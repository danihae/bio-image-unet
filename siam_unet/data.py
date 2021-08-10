import os, glob, re, shutil
import subprocess
from sys import platform
import time
from numpy.lib.function_base import diff
import random

import tifffile
import numpy as np
from skimage import morphology, transform
from barbar import Bar
from tifffile.tifffile import TiffFile
from tqdm import tqdm as tqdm

import torch
from torch import nn as nn, flatten
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from albumentations import (
    ShiftScaleRotate, GaussNoise,
    RandomBrightnessContrast, Flip, Compose, RandomRotate90)
from .losses import logcoshDiceLoss, BCEDiceLoss

from .helpers.util import write_info_file
from .helpers.md5sum import md5sum, md5sum_folder
import cv2

import wandb

class DataProcess(Dataset):
    def __init__(self, source_dir, file_ext='.tif', dim_out=(256, 256), aug_factor=10, data_path='./data/',
                 dilate_mask=0, val_split=0.2, invert=False, skeletonize=False, create=True, clip_thres=(0.2, 99.8),
                 shiftscalerotate=(0, 0, 0), rescale=None):
        """Processes input images and creates a dataloader for Siam_UNet

        Args:
            source_dir (list of 2): [concatenated_image_directory, label_directory]
            file_ext (str, optional): Defaults to '.tif'. Anything other than '.tif' has not been tested.
            dim_out (tuple, optional): The dimensions of the final training images. Defaults to (256, 256).
            aug_factor (int, optional): Augmentation factor. This is how many times each image/label pair is amplified. Defaults to 10.
            data_path (str, optional): In which directory to save the training images and masks. Defaults to './data/'.
            dilate_mask (int, optional): Mask dilation factor. Defaults to 0. View the references of `dilate_mask` in this class to understand what this factor does and customize the method of dilation.
            val_split (float, optional): What proportion of augmented image/label pairs to be used as validation dataset. Defaults to 0.2.
            invert (bool, optional): Whether to invert the labels. Defaults to False.
            skeletonize (bool, optional): Whether to skeletonize the labels. Defaults to False.
            create (bool, optional): Whether to create this dataset. Defaults to True. If set to false, data is loaded from the disk from `data_path` and used directly for the dataloader
            clip_thres (tuple, optional): Uses numpy.clip() to clip the data in the training image. Defaults to (0.2, 99.8).
            shiftscalerotate (tuple, optional): Parameter passed albumentation.ShiftScaleRotate(). See documentation fo that function. Usage of this function is not recommended because of bad results. Defaults to (0, 0, 0).
            rescale (float, optional): Parameter `scale` passed to skimage.transform.rescale() for each image/mask pair. Usage of this function is not recommended because of bad results. Defaults to None.
        """
        self.source_dir = source_dir
        self.file_ext = file_ext
        self.create = create
        self.data_path = data_path
        self.dim_out = dim_out
        self.skeletonize = skeletonize
        self.invert = invert
        self.clip_thres = clip_thres
        self.aug_factor = aug_factor
        self.shiftscalerotate = shiftscalerotate
        self.rescale = rescale
        self.dilate_mask = dilate_mask
        self.val_split = val_split
        self.mode = 'train'

        self.make_dirs()

        if create:
            self.move_and_edit()
            self.merge_images()
            self.split()
            if self.aug_factor is not None:
                self.augment()

    def make_dirs(self):
        self.image_path = self.data_path + '/image/'
        self.prev_image_path = self.data_path + '/prev_image/'
        self.mask_path = self.data_path + '/mask/'
        self.npy_path = self.data_path + '/npydata/'
        self.merge_path = self.data_path + '/merge/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.split_image_path = self.data_path + '/split/image/'
        self.split_prev_image_path = self.data_path + '/split/prev_image/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.aug_image_path = self.data_path + '/augmentation/aug_image/'
        self.aug_mask_path = self.data_path + '/augmentation/aug_mask/'
        self.aug_prev_image_path = self.data_path + '/augmentation/aug_prev_image/'

        # delete old files
        if self.create:
            try:
                shutil.rmtree(self.data_path)
            except:
                pass
        # make folders
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.npy_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_image_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_image_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)
        os.makedirs(self.aug_prev_image_path, exist_ok=True)
        os.makedirs(self.prev_image_path, exist_ok=True)
        os.makedirs(self.split_prev_image_path, exist_ok=True)

    def move_and_edit(self):
        # create image data
        files_image = glob.glob(self.source_dir[0] + '*' + self.file_ext)
        for file_i in files_image:
            img_i = tifffile.imread(file_i)
            # clip and normalize (0,255)
            img_i_nan = np.copy(img_i).astype('float32')
            img_i_nan[img_i == 0] = np.nan
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i_nan, self.clip_thres[0]),
                            a_max=np.percentile(img_i_nan, self.clip_thres[1]))
            img_i = img_i - np.min(img_i)
            img_i = img_i / np.max(img_i) * 255
            img_i[np.isnan(img_i_nan)] = 0
            if self.rescale is not None:
                img_i = transform.rescale(img_i, self.rescale)
            img_i = img_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            # split the input image into two
            img_width = int(img_i.shape[1]/2)
            prev_img = img_i[:, 0:img_width]
            infer_img = img_i[:, img_width:]
            tifffile.imsave(self.prev_image_path + save_i + '.tif', prev_img)
            tifffile.imsave(self.image_path + save_i + '.tif', infer_img)

    
        # create masks
        files_mask = glob.glob(self.source_dir[1] + '*' + self.file_ext)
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            mask_i = cv2.imread(file_i, cv2.IMREAD_GRAYSCALE)
            if self.rescale is not None:
                mask_i = transform.rescale(mask_i, self.rescale) * 255
                mask_i[mask_i < 255] = 0

            if mask_i.shape[1] != img_width:
                print(f"Mask width {mask_i.shape[1]} doesn't match up with image width {img_width}. Have concatenated the input image with its previous frame?")
                raise IOError # mask width mismatch

            if self.skeletonize:
                mask_i[mask_i > 1] = 1
                mask_i = 1 - mask_i
                mask_i = morphology.skeletonize(mask_i)
                mask_i = 255 * (1 - mask_i)
            if self.dilate_mask > 0:
                mask_i = morphology.erosion(mask_i, morphology.disk(self.dilate_mask))
            elif self.dilate_mask < 0:
                mask_i = morphology.dilation(mask_i, morphology.disk(-self.dilate_mask))
            if self.invert:
                mask_i = 255 - mask_i
            mask_i = mask_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            cv2.imwrite(self.mask_path + save_i + '.tif', mask_i)

    def merge_images(self):
        self.mask_files = glob.glob(self.data_path + '/mask/*.tif')
        self.image_files = glob.glob(self.data_path + '/image/*.tif')
        self.prev_image_files = glob.glob(self.data_path + '/prev_image/*.tif')

        if len(self.mask_files) != len(self.image_files):
            raise ValueError('Number of ground truth does not match number of image stacks')

        for i, file_i in enumerate(self.mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            image_i = tifffile.imread(self.data_path + '/image/' + basename_i)
            prev_image_i = tifffile.imread(self.data_path + '/prev_image/' + basename_i)
            merge = np.zeros((mask_i.shape[0], mask_i.shape[1], 3))
            merge[:, :, 0] = mask_i
            merge[:, :, 1] = image_i
            merge[:, :, 2] = prev_image_i
            merge = merge.astype('uint8')
            tifffile.imsave(self.merge_path + str(i) + '.tif', merge)

    def split(self):
        self.merges = glob.glob(self.merge_path + '*.tif')
        n = 0
        for i in range(len(self.merges)):
            merge = tifffile.imread(self.merge_path + str(i) + '.tif')
            dim_in = merge.shape
            # padding if dim_in < dim_out
            x_gap = max(0, self.dim_out[0] - dim_in[0])
            y_gap = max(0, self.dim_out[1] - dim_in[1])
            merge = np.pad(merge, ((0, x_gap), (0, y_gap), (0, 0)), 'reflect')
            # number of patches in x and y
            dim_in = merge.shape
            N_x = int(np.ceil(dim_in[0] / self.dim_out[0]))
            N_y = int(np.ceil(dim_in[1] / self.dim_out[1]))
            # starting indices of patches
            X_start = np.linspace(0, dim_in[0] - self.dim_out[0], N_x).astype('int16')
            Y_start = np.linspace(0, dim_in[1] - self.dim_out[1], N_y).astype('int16')
            for j in range(N_x):
                for k in range(N_y):
                    patch_ij = merge[X_start[j]:X_start[j] + self.dim_out[0], Y_start[k]:Y_start[k] + self.dim_out[1],
                               :]
                    mask_ij = patch_ij[:, :, 0]
                    image_ij = patch_ij[:, :, 1]
                    prev_image_ij = patch_ij[:, :, 2]
                    tifffile.imsave(self.split_merge_path + f'{n}.tif', patch_ij)
                    tifffile.imsave(self.split_mask_path + f'{n}.tif', mask_ij)
                    tifffile.imsave(self.split_image_path + f'{n}.tif', image_ij)
                    tifffile.imsave(self.split_prev_image_path + f'{n}.tif', prev_image_ij)
                    n += 1

    def augment(self, p=0.8):
        aug_pipeline = Compose([
            Flip(),
            RandomRotate90(p=1.0),
            ShiftScaleRotate(self.shiftscalerotate[0], self.shiftscalerotate[1], self.shiftscalerotate[2]),
            GaussNoise(var_limit=(30, 0), p=0.3),
            RandomBrightnessContrast(p=0.5),
        ],
            p=p)

        patches_image = glob.glob(self.split_image_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')
        patches_prev_image = glob.glob(self.split_prev_image_path + '*.tif')
        n_patches = len(patches_image)
        k = 0
        for i in range(n_patches):
            image_i = tifffile.imread(patches_image[i])
            mask_i = tifffile.imread(patches_mask[i])
            prev_image_i = tifffile.imread(patches_prev_image[i])

            data_i = {'image': image_i, 'mask': mask_i, 'prev_image': prev_image_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            imgs_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])
            prev_image_aug_i = np.asarray([data_aug_i[j]['prev_image'] for j in range(self.aug_factor)])

            for j in range(self.aug_factor):
                tifffile.imsave(self.aug_image_path + f'{k}.tif', imgs_aug_i[j])
                tifffile.imsave(self.aug_mask_path + f'{k}.tif', masks_aug_i[j])
                tifffile.imsave(self.aug_prev_image_path + f'{k}.tif', prev_image_aug_i[j])
                k += 1
        print(f'Number of training images: {k}')

    def __len__(self):
        if self.aug_factor is not None:
            return len(os.listdir(self.aug_image_path))
        else:
            return len(os.listdir(self.split_image_path))

    def __getitem__(self, idx):
        imgname = str(idx) + '.tif'
        midname = os.path.basename(imgname)
        if self.aug_factor is not None:
            image_0 = tifffile.imread(self.aug_image_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.aug_mask_path + midname).astype('float32') / 255
            prev_image_0 = tifffile.imread(self.aug_prev_image_path + midname).astype('float32') / 255
        else:
            image_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
            prev_image_0 = tifffile.imread(self.split_prev_image_path + midname).astype('float32') / 255
        image = torch.from_numpy(image_0)
        mask = torch.from_numpy(mask_0)
        prev_image = torch.from_numpy(prev_image_0)
        del image_0, mask_0, prev_image_0
        sample = {'image': image, 'mask': mask, 'prev_image': prev_image}
        return sample
