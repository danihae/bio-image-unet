import glob
import os
import shutil
from skimage import morphology
from albumentations import (
    ShiftScaleRotate, GaussNoise,
    RandomBrightnessContrast, Flip, Compose, RandomRotate90)

import numpy as np
import tifffile
import torch
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
    dim_out : Tuple[int, int]
        Resize dimensions of volumes for training
    aug_factor : int
        Factor of volume augmentation
    data_path : str
        Base path of temporary directories for training data
    dilate_mask
        Radius of binary dilation of masks [-2, -1, 0, 1, 2]
    dilate_kernel : str
        Dilation kernel ('disk' or 'square')
    add_patch : int
        Add additional patch for splitting volumes into patches with more overlapping patches
    val_split : float
        Validation split for training
    invert : bool
        If True, greyscale binary labels is inverted
    skeletonize : bool
        If True, binary labels are skeletonized
    clip_threshold : Tuple[float, float]
        Clip thresholds for intensity normalization of volumes
    shiftscalerotate : [float, float, float]
        Shift, scale and rotate during augmentation
    noise_amp : float
        Amplitude of Gaussian noise for augmentation
    brightness_contrast : Tuple[float, float]
        Adapt brightness and contrast of volumes during augmentation
    rescale : float, optional
        Rescale all volumes and labels by factor rescale
    create : bool, optional
        If False, existing data set in data_path is used
    """

    def __init__(self, source_dir, dim_out=(128, 128, 128), aug_factor=10, data_path='../data/', dilate_mask=0, dilate_kernel='disk', add_patch=0,
                 val_split=0.2, invert=False, skeletonize=False, clip_threshold=(0.2, 99.8), shiftscalerotate=(0, 0, 0),
                 noise_amp=10, brightness_contrast=(0.25, 0.25), create=True):

        self.source_dir = source_dir
        self.create = create
        self.data_path = data_path
        self.dim_out = dim_out
        self.skeletonize = skeletonize
        self.invert = invert
        self.clip_threshold = clip_threshold
        self.add_patch = add_patch
        self.aug_factor = aug_factor
        self.shiftscalerotate = shiftscalerotate
        self.brightness_contrast = brightness_contrast
        self.noise_amp = noise_amp
        self.dilate_mask = dilate_mask
        self.dilate_kernel = dilate_kernel
        self.val_split = val_split
        self.mode = 'train'

        self.__make_dirs()
        if create:
            self.__move_and_edit()
            self.__merge_volumes()
            self.__split()
            if self.aug_factor is not None:
                self.__augment()

    def __make_dirs(self):
        self.volume_path = self.data_path + '/volume/'
        self.mask_path = self.data_path + '/mask/'
        self.merge_path = self.data_path + '/merge/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.split_volume_path = self.data_path + '/split/volume/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.aug_volume_path = self.data_path + '/augmentation/aug_volume/'
        self.aug_mask_path = self.data_path + '/augmentation/aug_mask/'

        # delete old files
        if self.create and os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

        # make folders
        os.makedirs(self.volume_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_volume_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_volume_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)

    def __move_and_edit(self):
        # create volume data
        files_volume = glob.glob(self.source_dir[0] + '*')
        for file_i in files_volume:
            vol_i = tifffile.imread(file_i).astype(np.float32)
            # clip and normalize (0,255)
            vol_i = np.clip(vol_i, a_min=np.percentile(vol_i, self.clip_threshold[0]),
                            a_max=np.percentile(vol_i, self.clip_threshold[1]))
            vol_i = (vol_i - np.nanmin(vol_i)) / (np.nanmax(vol_i) - np.nanmin(vol_i)) * 255
            vol_i = vol_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imwrite(self.volume_path + save_i + '.tif', vol_i)
        # create masks
        files_mask = glob.glob(self.source_dir[1] + '*')
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            mask_i = tifffile.imread(file_i)
            for j, ch_j in enumerate(mask_i):  # iterate frames / slices
                if self.skeletonize:
                    ch_j = ch_j > 1
                    ch_j = morphology.skeletonize(ch_j) * 255
                if self.dilate_kernel == 'disk':
                    kernel = morphology.disk
                elif self.dilate_kernel == 'square':
                    kernel = morphology.square
                else:
                    raise ValueError(f'Dilate kernel {self.dilate_kernel} unknown!')
                if self.dilate_mask > 0 and self.dilate_mask:
                    ch_j = morphology.erosion(ch_j, kernel(self.dilate_mask))
                elif self.dilate_mask < 0:
                    ch_j = morphology.dilation(ch_j, kernel(-self.dilate_mask))
                if self.invert:
                    ch_j = 255 - ch_j
                mask_i[j] = ch_j
            mask_i = mask_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imwrite(self.mask_path + save_i + '.tif', mask_i.astype('int8'))

    def __merge_volumes(self):
        self.mask_files = glob.glob(self.data_path + '/mask/*.tif')
        self.volume_files = glob.glob(self.data_path + '/volume/*.tif')

        if len(self.mask_files) != len(self.volume_files):
            raise ValueError('Number of ground truth does not match number of volume stacks')

        for i, file_i in enumerate(self.mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            vol_i = tifffile.imread(self.data_path + '/volume/' + basename_i)
            merge = np.zeros((mask_i.shape[0], mask_i.shape[1], mask_i.shape[2], 2), dtype='uint8')
            merge[:, :, :, 0] = vol_i
            merge[:, :, :, 1] = mask_i
            tifffile.imwrite(self.merge_path + str(i) + '.tif', merge)

    def __split(self):
        self.merges = glob.glob(self.merge_path + '*.tif')
        n = 0
        for i in range(len(self.merges)):
            merge = tifffile.imread(self.merge_path + str(i) + '.tif')
            dim_in = merge.shape
            # padding if dim_in < dim_out
            z_gap = max(0, self.dim_out[0] - dim_in[0])
            x_gap = max(0, self.dim_out[1] - dim_in[1])
            y_gap = max(0, self.dim_out[2] - dim_in[2])
            merge = np.pad(merge, ((0, z_gap), (0, x_gap), (0, y_gap), (0, 0)), 'reflect')
            # number of patches in x and y
            dim_in = merge.shape
            N_z = int(np.ceil(dim_in[0] / self.dim_out[0]))
            N_x = int(np.ceil(dim_in[1] / self.dim_out[1]))
            N_y = int(np.ceil(dim_in[2] / self.dim_out[2]))
            N_x += self.add_patch if N_z > 1 else 0
            N_x += self.add_patch if N_x > 1 else 0
            N_y += self.add_patch if N_y > 1 else 0
            # starting indices of patches
            Z_start = np.linspace(0, dim_in[0] - self.dim_out[0], N_z).astype('int16')
            X_start = np.linspace(0, dim_in[1] - self.dim_out[1], N_x).astype('int16')
            Y_start = np.linspace(0, dim_in[2] - self.dim_out[2], N_y).astype('int16')
            for j in range(N_z):
                for k in range(N_x):
                    for l in range(N_y):
                        patch_ij = merge[Z_start[j]:Z_start[j] + self.dim_out[0],
                                   X_start[k]: X_start[k] + self.dim_out[1],
                                   Y_start[l]: Y_start[l] + self.dim_out[2], :]
                        volume_ij = patch_ij[:, :, :, 0]
                        mask_ij = patch_ij[:, :, :, 1]

                        tifffile.imwrite(self.split_merge_path + f'{n}.tif', patch_ij)
                        tifffile.imwrite(self.split_mask_path + f'{n}.tif', mask_ij)
                        tifffile.imwrite(self.split_volume_path + f'{n}.tif', volume_ij)
                        n += 1

    def __augment(self, p=0.8):
        aug_pipeline = Compose(transforms=[
            Flip(),
            RandomRotate90(p=1.0),
            ShiftScaleRotate(self.shiftscalerotate[0], self.shiftscalerotate[1], self.shiftscalerotate[2]),
            GaussNoise(var_limit=(self.noise_amp, self.noise_amp), p=0.3),
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=0.5),
        ],
            p=p)

        patches_volume = glob.glob(self.split_volume_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')

        n_patches = len(patches_volume)
        k = 0
        for i in range(n_patches):
            volume_i = tifffile.imread(patches_volume[i])
            mask_i = tifffile.imread(patches_mask[i])
            volume_i = volume_i.transpose(1, 2, 0)
            mask_i = mask_i.transpose(1, 2, 0)

            data_i = {'image': volume_i, 'mask': mask_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            vols_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])

            for j in range(self.aug_factor):
                tifffile.imwrite(self.aug_volume_path + f'{k}.tif', vols_aug_i[j].transpose(2, 0, 1))
                tifffile.imwrite(self.aug_mask_path + f'{k}.tif', masks_aug_i[j].transpose(2, 0, 1))
                k += 1
        print(f'Number of training volumes: {k}')

    def __len__(self):
        if self.aug_factor is not None:
            return len(os.listdir(self.aug_volume_path))
        else:
            return len(os.listdir(self.split_volume_path))

    def __getitem__(self, idx):
        volname = str(idx) + '.tif'
        midname = os.path.basename(volname)
        if self.aug_factor is not None:
            volume_0 = tifffile.imread(self.aug_volume_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.aug_mask_path + midname).astype('float32') / 255
        else:
            volume_0 = tifffile.imread(self.split_volume_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.split_volume_path + midname).astype('float32') / 255
        volume = torch.from_numpy(volume_0)
        mask = torch.from_numpy(mask_0)
        del volume_0, mask_0
        sample = {'volume': volume, 'mask': mask}
        return sample
