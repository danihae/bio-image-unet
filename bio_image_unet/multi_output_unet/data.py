import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np
import tifffile
import torch
from albumentations import (
    Blur, GaussNoise, ShotNoise,
    RandomBrightnessContrast, Compose, RandomScale, RandomCrop, PadIfNeeded)
from scipy.ndimage import rotate
from torch.utils.data import Dataset
from tqdm import tqdm


class DataProcess(Dataset):
    """Class for training data generation (processing, splitting, augmentation)"""

    def __init__(self,
                 image_dir: str,
                 target_dirs: List[str],
                 target_types: dict[str, str],
                 data_dir: str = '../data/',
                 dim_out: Tuple[int, int] = (256, 256),
                 in_channels: int = 1,
                 out_channels: int = 1,
                 add_tile: int = 0,
                 nan_to_val: float = 0,
                 val_split: float = 0.2,
                 clip_threshold: Tuple[float, float] = (0., 99.99),
                 aug_factor: float = 2,
                 gauss_noise_lims: Tuple[float, float] = (0.01, 0.1),
                 shot_noise_lims: Tuple[float, float] = (0.001, 0.01),
                 brightness_contrast: Tuple[float, float] = (0.1, 0.1),
                 blur_limit: Tuple[int, int] = (3, 5),
                 random_rotate: bool = True,
                 scale_limit: Tuple[float, float] = (0, 0),
                 create: bool = True,
                 file_filter: Optional[Callable[[str], bool]] = None):
        """
        Create training data object for network training

        1) Create folder structure for training data
        2) Move and preprocess training images
        3) Split input images into tiles
        4) Augment training data
        5) Create object of PyTorch Dataset class for training

        Parameters
        ----------
        image_dir : str
            Directory with training images
        target_dirs : List[str]
            List of directories with targets
        target_types : dict
            Dictionary of target types (e.g. {'feature1': 'mask', 'feature2': 'mask'})
        dim_out : Tuple[int, int]
            Resize dimensions of images for training
        data_dir : str
            Base dir of temporary directories for training data
        in_channels : int
            Number of input channels of training data
        out_channels : int
            Number of output channels of training data
        add_tile : int
            Add additional tile for splitting images into tiles with more overlapping tiles
        nan_to_val: float
            Value to use for missing (NaN) pixels in targets
        val_split : float
            Validation split for training
        clip_threshold : Tuple[float, float]
            Clip thresholds for intensity normalization of images
        aug_factor : float
            Augmentation factor, number of patches per image is determined by
            n_pixels_image/n_pixels_patch * aug_factor.
        gauss_noise_lims : Tuple[float, float]
            Limits of Gaussian noise for image augmentation
        shot_noise_lims : Tuple[float, float]
            Limits of Shot noise for image augmentation
        brightness_contrast : Tuple[float, float]
            Adapt brightness and contrast of images during augmentation
        blur_limit : Tuple[float, float]
            Limits of blur (min, max) for image augmentation
        random_rotate : bool
            Whether to randomly rotate images during augmentation
        scale_limit : Tuple[float, float]
            Limits of scale (min, max) for image augmentation
        create : bool, optional
            If False, existing data set in data_path is used
        file_filter : function, optional
            Function with filename as arg that returns bool values indicating whether to include files in the dataset

        Methods
        -------
        __len__()
            Returns the total number of samples.
        __getitem__(idx)
            Retrieves the sample at the specified index.
        """
        self.image_dir = image_dir
        self.target_dirs = target_dirs
        self.target_keys = [os.path.basename(os.path.normpath(_dir)) for _dir in target_dirs]
        self.target_types = target_types
        self.data_dir = data_dir
        self.data = []  # empty list for data dicts
        self.create = create
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_out = dim_out
        self.nan_to_val = nan_to_val
        self.clip_threshold = clip_threshold
        self.add_tile = add_tile
        self.aug_factor = aug_factor
        self.brightness_contrast = brightness_contrast
        self.gauss_noise_lims = gauss_noise_lims
        self.shot_noise_lims = shot_noise_lims
        self.blur_limit = blur_limit
        self.random_rotate = random_rotate
        self.scale_limit = scale_limit
        self.file_filter = file_filter

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
            self.__augment()

    def __read_and_edit(self):
        # Find all TIFF files in image_dir
        image_path = Path(self.image_dir)
        files_image = [str(file) for ext in ['*.tif', '*.tiff'] for file in image_path.glob(ext)]

        if self.file_filter:
            files_image = [file for file in files_image if self.file_filter(file)]

        # Read and normalize images
        for file_img_i in files_image:
            img_i = tifffile.imread(file_img_i)
            basename_i = os.path.basename(file_img_i)

            # Create dict for image and targets
            data_i = {}

            # Clip and normalize
            img_i = img_i.astype(np.float32)
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i, self.clip_threshold[0]),
                            a_max=np.percentile(img_i, self.clip_threshold[1]))
            img_i = (img_i - np.nanmin(img_i)) / (np.nanmax(img_i) - np.nanmin(img_i))

            # Add image to dict
            data_i['image'] = img_i

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

            # Add data to the list only if all targets exist and check shape consistency
            if all_targets_exist:
                shapes = []
                for key in data_i:
                    shapes.append(data_i[key].shape)
                    if len(shapes) > 1 and shapes[0] != shapes[-1]:
                        raise ValueError(f"File: {file_img_i}. Shape mismatch: {key} has shape {shapes[-1]} but expected {shapes[0]}")
                self.data.append(data_i)

    def __augment(self):
        target_types = {key: self.target_types[key] for key in self.target_keys}
        aug_pipeline = Compose(transforms=[
            RandomScale(scale_limit=self.scale_limit, p=0.75,
                        interpolation=cv2.INTER_NEAREST,
                        mask_interpolation=cv2.INTER_NEAREST),
            Blur(blur_limit=self.blur_limit, p=0.25),
            PadIfNeeded(min_height=self.dim_out[0], min_width=self.dim_out[1],
                        border_mode=cv2.BORDER_WRAP, position='bottom_left'),
            RandomCrop(height=self.dim_out[0], width=self.dim_out[1], p=1),
            ShotNoise(scale_range=self.shot_noise_lims, p=0.25),
            GaussNoise(std_range=self.gauss_noise_lims, p=0.25),
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=0.5),
        ], additional_targets=target_types)

        def chw_to_hwc(x):
            if len(x.shape) == 3:  # Only convert if it has channels
                return np.transpose(x, (1, 2, 0))
            return x

        def hwc_to_chw(x):
            if len(x.shape) == 3:  # Only convert if it has channels
                return np.transpose(x, (2, 0, 1))
            return x

        def rotate_array(x, angle, order=1):
            # Convert bool arrays to float32 first
            if x.dtype == bool or x.dtype == np.bool_:
                x = x.astype(np.float32)
                xmin, xmax = 0, 1
                needs_clip = True
            elif np.nanmin(x) >= 0 and np.nanmax(x) <= 1:  # for probability masks with [0, 1] range
                xmin, xmax = np.nanmin(x), np.nanmax(x)
                needs_clip = True
            else:
                xmin, xmax = None, None
                needs_clip = False

            if np.any(np.isnan(x)):
                nan_mask = np.isnan(x)
                x_filled = np.where(nan_mask, 0, x)
                rotated = rotate(x_filled, angle, reshape=False, mode='grid-wrap', order=order)
                rotated_mask = rotate(nan_mask.astype(np.uint8), angle, reshape=False,
                                      mode='grid-wrap', order=order) > 0.5
                rotated = rotated.astype(np.float32)
                rotated[rotated_mask] = np.nan
            else:
                rotated = rotate(x, angle, reshape=False, mode='grid-wrap', order=order)
                rotated = rotated.astype(np.float32)

            # Clip boolean-originated and [0, 1] range arrays
            if needs_clip:
                rotated = np.clip(rotated, xmin, xmax)

            return rotated

        def rotate_array_90(x, factor):
            if len(x.shape) == 3 and x.shape[0] < x.shape[1]:  # CHW format
                return np.rot90(x, factor, axes=(1, 2))
            else:  # HW or HWC format
                return np.rot90(x, factor)

        running_number = 0

        for patch_data in tqdm(self.data):
            # Convert image and targets, only if they have channels
            image_i = patch_data['image'].astype('float32')
            targets_i = {key: patch_data[key].astype('float32')
                         for key in patch_data if key != 'image'}

            aug_factor = max(int((image_i.shape[0] * image_i.shape[1]) /
                             (self.dim_out[0] * self.dim_out[1]) * self.aug_factor), 2)

            for _ in range(aug_factor):
                aug_image = image_i.copy()
                aug_targets = targets_i.copy()
                if self.random_rotate:
                    if random.random() < 0.5:
                        # Apply random rotation with any angle
                        angle = np.random.uniform(0, 360)
                        aug_image = rotate_array(aug_image, angle, order=0)

                        for key in aug_targets:
                            aug_targets[key] = rotate_array(aug_targets[key], angle, order=3)
                            if 'orientation' in key:
                                aug_targets[key] = (aug_targets[key] - np.radians(angle)) % (2 * np.pi)
                    else:
                        # Apply random rotation by 90 degree
                        factor = np.random.randint(0, 3)
                        aug_image = rotate_array_90(aug_image, factor=factor)
                        for key in aug_targets:
                            if 'orientation' in key:
                                # Adjust orientation values and rotate
                                aug_targets[key] = (aug_targets[key] - (np.pi / 2 * factor)) % (2 * np.pi)
                            aug_targets[key] = rotate_array_90(aug_targets[key], factor=factor)

                aug_data_hwc = {'image': np.clip(aug_image, 0, 1)}
                aug_targets_hwc = {key: chw_to_hwc(aug_targets[key]) for key in aug_targets}
                aug_data_hwc.update(aug_targets_hwc)

                augmented = aug_pipeline(**aug_data_hwc)

                # Convert back to CHW format only if needed
                aug_image = hwc_to_chw(augmented['image'])
                aug_targets = {key: hwc_to_chw(augmented[key])
                               for key in targets_i}

                # Save augmented image
                image_dir = os.path.join(self.data_dir, 'image')
                os.makedirs(image_dir, exist_ok=True)
                tifffile.imwrite(os.path.join(image_dir, f'image_{running_number}.tif'),
                                 aug_image)

                # Save augmented targets
                for target_key, target_value in aug_targets.items():
                    target_dir = os.path.join(self.data_dir, target_key)
                    os.makedirs(target_dir, exist_ok=True)
                    tifffile.imwrite(os.path.join(target_dir,
                                                  f'{target_key}_{running_number}.tif'),
                                     target_value)

                running_number += 1

        print(f'Augmentation completed for {running_number} patches.')

    def __len__(self):
        # Count the number of image files in the 'image' directory within the data path
        image_dir = os.path.join(self.data_dir, 'image')
        return len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])

    def __getitem__(self, idx):
        # Construct the file name for the image
        img_name = f'image_{idx}.tif'

        # Read the image from the 'image' directory
        image_path = os.path.join(self.data_dir, 'image', img_name)
        image_0 = tifffile.imread(image_path)

        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(image_0)

        # Initialize the sample dictionary with the image
        sample = {'image': image}

        # Iterate over each target directory and read the corresponding target file
        for target_dir in self.target_dirs:
            target_key = os.path.basename(os.path.normpath(target_dir))
            target_file = f'{target_key}_{idx}.tif'
            target_path = os.path.join(self.data_dir, target_key, target_file)
            if os.path.exists(target_path):
                target_data = tifffile.imread(target_path).astype('float32')
                # convert to vector field if 'orientation'
                if target_key == 'orientation':
                    target_data = np.stack([np.cos(target_data), np.sin(target_data)])
                # set NaNs to val
                target_data[np.isnan(target_data)] = self.nan_to_val
                sample[target_key] = torch.from_numpy(target_data)
            else:
                raise FileNotFoundError(f"Target file {target_path} not found.")

        # Return the sample as a dictionary
        return sample
