import os
import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tifffile
import torch

from albumentations import (ShiftScaleRotate, GaussNoise, ShotNoise, RandomCrop3D, RandomBrightnessContrast, Compose,
                            Blur)
from torch.utils.data import Dataset


class DataProcess(Dataset):
    """
    Class for training data generation (processing, splitting, augmentation)

    Create training data object for network training

    1) Create folder structure for training data
    2) Move and preprocess training volumes
    3) Split input volumes into patches
    4) Augment training data
    5) Create object of PyTorch Dataset class for training

    Parameters
    ----------
    source_dir : Tuple[str, str]
        Path of training data [volumes, labels]. volumes need to be tif files.
    dim_out : Tuple[int, int, int]
        Resize dimensions of volumes for training
    data_path : str
        Base path of temporary directories for training data
    clip_threshold : Tuple[float, float]
        Clip thresholds for intensity normalization of volumes
    add_patch : int
        Add additional patch for splitting volumes into patches with more overlapping patches
    val_split : float
        Validation split for training
    aug_factor : int
        Factor of volume augmentation
    shiftscalerotate : Tuple[float, float, float]
        Shift, scale and rotate during augmentation
    gauss_noise_lims : Tuple[float, float]
        Limits of Gaussian noise for volume augmentation
    shot_noise_lims : Tuple[float, float]
        Limits of Shot noise for volume augmentation
    brightness_contrast : Tuple[float, float]
        Adapt brightness and contrast of volumes during augmentation
    blur_limit : Tuple[int, int]
        Blur limit range for augmentation
    create : bool
        If False, existing data set in data_path is used
    """

    def __init__(self,
                 volume_dir: str,
                 target_dirs: List[str],
                 data_dir: str = '../data/',
                 dim_out: Tuple[int, int, int] = (128, 128, 128),
                 in_channels: int = 1,
                 add_tile: int = 0,
                 nan_to_val: float = 0,
                 val_split: float = 0.2,
                 clip_threshold: Tuple[float, float] = (0., 99.99),
                 aug_factor: int = 10,
                 scale_limit: Tuple[float, float] = (-0.75, 0),
                 rotate_limit: Tuple[float, float] = (0, 360),
                 gauss_noise_lims: Tuple[float, float] = (0.01, 0.1),
                 shot_noise_lims: Tuple[float, float] = (0.005, 0.01),
                 brightness_contrast: Tuple[float, float] = (0.1, 0.1),
                 blur_limit: Tuple[int, int] = (3, 7),
                 random_rotate: bool = True,
                 create: bool = True):

        self.volume_dir = volume_dir
        self.target_dirs = target_dirs
        self.target_keys = [os.path.basename(os.path.normpath(_dir)) for _dir in target_dirs]
        self.data_dir = data_dir
        self.data = []  # empty list for data dicts
        self.create = create
        self.in_channels = in_channels
        self.dim_out = dim_out
        self.nan_to_val = nan_to_val
        self.clip_threshold = clip_threshold
        self.add_tile = add_tile
        self.aug_factor = aug_factor
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.brightness_contrast = brightness_contrast
        self.gauss_noise_lims = gauss_noise_lims
        self.shot_noise_lims = shot_noise_lims
        self.blur_limit = blur_limit
        self.random_rotate = random_rotate

        self.val_split = val_split
        self.mode = 'train'

        if create:
            # Delete the existing data directory if it exists
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)

            # Create a new data directory
            os.makedirs(self.data_dir, exist_ok=True)

            # Proceed with data processing
            self.__read_and_edit()
            if self.aug_factor is not None:
                self.__augment()

    def __read_and_edit(self):
        # Find all TIFF files in volume_dir
        volume_path = Path(self.volume_dir)
        files_volume = [str(file) for ext in ['*.tif', '*.tiff'] for file in volume_path.glob(ext)]

        # Read and normalize volumes
        for file_img_i in files_volume:
            img_i = tifffile.imread(file_img_i)
            basename_i = os.path.basename(file_img_i)

            # Create dict for volume and targets
            data_i = {}

            # Clip and normalize
            img_i = img_i.astype(np.float32)
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i, self.clip_threshold[0]),
                            a_max=np.percentile(img_i, self.clip_threshold[1]))
            img_i = (img_i - np.nanmin(img_i)) / (np.nanmax(img_i) - np.nanmin(img_i))

            # Add volume to dict
            data_i['volume'] = img_i

            # Iterate through target folders and add targets to dict
            all_targets_exist = True
            for target_dir_j in self.target_dirs:
                file_target_j = os.path.join(target_dir_j, basename_i)
                if os.path.exists(file_target_j):
                    target_data = tifffile.imread(file_target_j)
                    target_key = os.path.basename(target_dir_j[:-1])  # Use the folder name as the target key
                    data_i[target_key] = target_data
                else:
                    print(f"Warning: Target file {file_target_j} does not exist.")
                    all_targets_exist = False
                    break

            # Add data to the list only if all targets exist
            if all_targets_exist:
                self.data.append(data_i)

    def __augment(self, p=0.8):
        # Define additional targets for masks
        additional_targets_3D = {key: 'mask3d' for key in self.target_keys}

        # Pipeline for 3D spatial transformations
        aug_pipeline_3D = Compose(
            transforms=[
                ShiftScaleRotate(shift_limit=0, scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, p=0.8),
                RandomCrop3D(size=self.dim_out),
            ],
            additional_targets=additional_targets_3D
        )

        # Pipeline for 2D intensity transformations (slice-by-slice)
        aug_pipeline = Compose(
            transforms=[
                RandomBrightnessContrast(
                    brightness_limit=self.brightness_contrast[0],
                    contrast_limit=self.brightness_contrast[1],
                    p=0.5
                ),
                Blur(blur_limit=self.blur_limit, p=0.3),
                ShotNoise(scale_range=self.shot_noise_lims, p=0.5),
                GaussNoise(std_range=self.gauss_noise_lims, p=0.5),
            ],
            p=p, additional_targets=additional_targets_3D,
        )

        running_number = 0

        # Iterate over data patches
        for patch_data in self.data:
            # Extract main volume and target masks
            volume_i = patch_data['volume']
            targets_i = {key: patch_data[key] for key in patch_data if key != 'volume'}

            # Prepare input data for 3D pipeline
            data_i = {'volume': volume_i}
            data_i.update(targets_i)

            for _ in range(self.aug_factor):
                # Apply 3D spatial transformations
                augmented_3D = aug_pipeline_3D(**data_i)

                # Extract augmented volume and targets from the first pipeline
                _aug_volume = augmented_3D['volume']
                _aug_targets = {key: augmented_3D[key] for key in targets_i}

                # Prepare input data for 2D pipeline (remap "volume" to "image")
                data_j = {'images': _aug_volume}
                data_j.update(_aug_targets)

                # Apply 2D intensity transformations
                augmented = aug_pipeline(**data_j)

                # Extract final augmented volume and targets
                aug_volume = augmented['images']
                aug_targets = {key: augmented[key] for key in targets_i}

                # Save augmented volume
                volume_dir = os.path.join(self.data_dir, 'volume')
                os.makedirs(volume_dir, exist_ok=True)
                tifffile.imwrite(
                    os.path.join(volume_dir, f'volume_{running_number}.tif'),
                    aug_volume
                )

                # Save augmented masks/targets
                for target_key, target_value in aug_targets.items():
                    target_dir = os.path.join(self.data_dir, target_key)
                    os.makedirs(target_dir, exist_ok=True)
                    tifffile.imwrite(
                        os.path.join(target_dir, f'{target_key}_{running_number}.tif'),
                        target_value
                    )

                running_number += 1

        print(f'Augmentation completed for {running_number} patches.')

    def __len__(self):
        # Count the number of volume files in the 'volume' directory within the data path
        volume_dir = os.path.join(self.data_dir, 'volume')
        return len([name for name in os.listdir(volume_dir) if os.path.isfile(os.path.join(volume_dir, name))])

    def __getitem__(self, idx):
        # Construct the file name for the volume
        img_name = f'volume_{idx}.tif'

        # Read the volume from the 'volume' directory
        volume_path = os.path.join(self.data_dir, 'volume', img_name)
        volume_0 = tifffile.imread(volume_path)

        # Convert the volume to a PyTorch tensor
        volume = torch.from_numpy(volume_0)

        # Initialize the sample dictionary with the volume
        sample = {'volume': volume}

        # Iterate over each target directory and read the corresponding target file
        for target_dir in self.target_dirs:
            target_key = os.path.basename(os.path.normpath(target_dir))
            target_file = f'{target_key}_{idx}.tif'
            target_path = os.path.join(self.data_dir, target_key, target_file)
            if os.path.exists(target_path):
                target_data = tifffile.imread(target_path).astype('float32')
                if target_key == 'orientation':
                    target_data = np.stack([np.cos(target_data), np.sin(target_data)])
                # set NaNs to val
                target_data[np.isnan(target_data)] = self.nan_to_val
                sample[target_key] = torch.from_numpy(target_data)
            else:
                raise FileNotFoundError(f"Target file {target_path} not found.")

        # Return the sample as a dictionary
        return sample
