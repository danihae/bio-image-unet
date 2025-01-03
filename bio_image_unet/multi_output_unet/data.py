import os
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
from albumentations import (
    Blur, GaussNoise,
    RandomBrightnessContrast, Compose, RandomRotate90)
from torch.utils.data import Dataset


class DataProcess(Dataset):
    """Class for training data generation (processing, splitting, augmentation)"""

    def __init__(self, image_dir, target_dirs, target_types, data_dir='../data/', dim_out=(256, 256),
                 in_channels=1, out_channels=1, add_tile=0, nan_to_val=0,
                 val_split=0.2, clip_threshold=(0.1, 99.9), aug_factor=10,
                 noise_lims=(5, 25), brightness_contrast=(0.15, 0.15), blur_limit=(3, 10), create=True,
                 file_filter=None):
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
        aug_factor : int
            Factor of image augmentation
        noise_lims : float
            Limits of Gaussian noise for image augmentation
        brightness_contrast : Tuple[float, float]
            Adapt brightness and contrast of images during augmentation
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
        self.target_keys = [os.path.basename(os.path.normpath(dir)) for dir in target_dirs]
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
        self.noise_lims = noise_lims
        self.blur_limit = blur_limit
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
            self.__split()
            if self.aug_factor is not None:
                self.__augment()

    def __read_and_edit(self):
        # Find all TIFF files in image_dir
        image_path = Path(self.image_dir)
        files_image = [str(file) for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF'] for file in image_path.glob(ext)]

        if self.file_filter:
            files_image = [file for file in files_image if self.file_filter(file)]

        # Read and normalize images
        for file_img_i in files_image:
            img_i = tifffile.imread(file_img_i)
            basename_i = os.path.basename(file_img_i)

            # Create dict for image and targets
            data_i = {}

            # Clip and normalize (0,255)
            img_i = img_i.astype(np.float32)
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i, self.clip_threshold[0]),
                            a_max=np.percentile(img_i, self.clip_threshold[1]))
            img_i = (img_i - np.nanmin(img_i)) / (np.nanmax(img_i) - np.nanmin(img_i)) * 255
            img_i = img_i.astype('uint8')

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

            # Add data to the list only if all targets exist
            if all_targets_exist:
                self.data.append(data_i)

    def __split(self):
        self.data_split = []
        for data_i in self.data:
            dim_in = data_i['image'].shape

            # Calculate padding if dim_in < dim_out
            x_gap = max(0, self.dim_out[0] - dim_in[-2])
            y_gap = max(0, self.dim_out[1] - dim_in[-1])

            # Determine padding width based on image dimensions
            if len(dim_in) == 3:  # If image has channels (C, H, W)
                pad_width = ((0, 0), (x_gap // 2, x_gap - x_gap // 2), (y_gap // 2, y_gap - y_gap // 2))
            elif len(dim_in) == 2:  # If image is 2D (H, W)
                pad_width = ((x_gap // 2, x_gap - x_gap // 2), (y_gap // 2, y_gap - y_gap // 2))
            else:
                raise ValueError("Unsupported image dimensions")

            # Apply padding to the image
            padded_image = np.pad(data_i['image'], pad_width, mode='reflect')

            # Pad each target similarly
            padded_targets = {}
            for target_key, target_value in data_i.items():
                if target_key != 'image':
                    target_shape = target_value.shape
                    if len(target_shape) == 3:  # Assuming shape is (channels, height, width)
                        pad_width_target = ((0, 0), (x_gap // 2, x_gap - x_gap // 2), (y_gap // 2, y_gap - y_gap // 2))
                    elif len(target_shape) == 2:  # Assuming shape is (height, width)
                        pad_width_target = ((x_gap // 2, x_gap - x_gap // 2), (y_gap // 2, y_gap - y_gap // 2))
                    else:
                        raise ValueError("Unsupported target dimensions")
                    padded_targets[target_key] = np.pad(target_value, pad_width_target, mode='reflect')

            # Number of patches in x and y
            N_x = int(np.ceil(padded_image.shape[-2] / self.dim_out[0]))
            N_y = int(np.ceil(padded_image.shape[-1] / self.dim_out[1]))
            N_x += self.add_tile if N_x > 1 else 0
            N_y += self.add_tile if N_y > 1 else 0

            # Starting indices of patches
            X_start = np.linspace(0, padded_image.shape[-2] - self.dim_out[0], N_x).astype('int16')
            Y_start = np.linspace(0, padded_image.shape[-1] - self.dim_out[1], N_y).astype('int16')

            for j in range(N_x):
                for k in range(N_y):
                    # Create a new dictionary for each patch
                    patch_data = {}

                    # Extract image patch
                    if len(dim_in) == 3:
                        patch_image = padded_image[
                                      :,
                                      X_start[j]:X_start[j] + self.dim_out[0],
                                      Y_start[k]:Y_start[k] + self.dim_out[1]
                                      ]
                    else:
                        patch_image = padded_image[
                                      X_start[j]:X_start[j] + self.dim_out[0],
                                      Y_start[k]:Y_start[k] + self.dim_out[1]
                                      ]
                    patch_data['image'] = patch_image

                    # Extract target patches
                    for target_key, padded_target in padded_targets.items():
                        if len(padded_target.shape) == 3:  # Assuming shape is (channels, height, width)
                            patch_target = padded_target[
                                           :,
                                           X_start[j]:X_start[j] + self.dim_out[0],
                                           Y_start[k]:Y_start[k] + self.dim_out[1]
                                           ]
                        else:  # Assuming shape is (height, width)
                            patch_target = padded_target[
                                           X_start[j]:X_start[j] + self.dim_out[0],
                                           Y_start[k]:Y_start[k] + self.dim_out[1]
                                           ]
                        patch_data[target_key] = patch_target

                    # Add the patch data to the split list
                    self.data_split.append(patch_data)

    def __augment(self, p=0.8):
        target_types = {key: self.target_types[key] for key in self.target_keys}
        aug_pipeline = Compose(transforms=[
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=1),
            Blur(blur_limit=self.blur_limit, always_apply=False, p=0.3),
            GaussNoise(var_limit=self.noise_lims, p=0.5),
        ], p=p, additional_targets=target_types)

        running_number = 0

        def chw_to_hwc(x):
            if len(x.shape) == 3:  # Only convert if it has channels
                return np.transpose(x, (1, 2, 0))
            return x

        def hwc_to_chw(x):
            if len(x.shape) == 3:  # Only convert if it has channels
                return np.transpose(x, (2, 0, 1))
            return x

        def rotate_array(x, factor):
            if len(x.shape) == 3 and x.shape[0] < x.shape[1]:  # CHW format
                return np.rot90(x, factor, axes=(1, 2))
            else:  # HW or HWC format
                return np.rot90(x, factor)

        for patch_data in self.data_split:
            # Convert image and targets, only if they have channels
            image_i = chw_to_hwc(patch_data['image'])
            targets_i = {key: chw_to_hwc(patch_data[key])
                         for key in patch_data if key != 'image'}

            data_i = {'image': image_i}
            data_i.update(targets_i)

            for _ in range(self.aug_factor):
                augmented = aug_pipeline(**data_i)

                # Convert back to CHW format only if needed
                aug_image = hwc_to_chw(augmented['image'])
                aug_targets = {key: hwc_to_chw(augmented[key])
                               for key in targets_i}

                # Apply random rotation
                factor = np.random.randint(0, 3)
                aug_image = rotate_array(aug_image, factor=factor)

                for key in aug_targets:
                    if 'orientation' in key:
                        # Adjust orientation values and rotate
                        aug_targets[key] = (aug_targets[key] - (np.pi / 2 * factor)) % (2 * np.pi)
                    aug_targets[key] = rotate_array(aug_targets[key], factor=factor)

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

        print(f'Augmentation completed for {len(self.data_split) * self.aug_factor} patches.')

    def __len__(self):
        # Count the number of image files in the 'image' directory within the data path
        image_dir = os.path.join(self.data_dir, 'image')
        return len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])

    def __getitem__(self, idx):
        # Construct the file name for the image
        img_name = f'image_{idx}.tif'

        # Read the image from the 'image' directory
        image_path = os.path.join(self.data_dir, 'image', img_name)
        image_0 = tifffile.imread(image_path).astype('float32') / 255

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