import glob
import os
import shutil
from skimage import morphology
from albumentations import (
    ShiftScaleRotate, Blur, MultiplicativeNoise,
    RandomBrightnessContrast, Flip, Compose, RandomRotate90)

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class DataProcess(Dataset):
    """
    Class for training data generation (processing, splitting, augmentation)

    Creates training data object for network training

    1) Create folder structure for training data
    2) Move and preprocess training images
    3) Split input images into tiles
    4) Augment training data
    5) Create object of PyTorch Dataset class for training

    Parameters
    ----------
    source_dir : Tuple[str, str]
        Path of training data [images, labels]. Images need to be tif files.
    dim_out : Tuple[int, int]
        Resize dimensions of images for training
    aug_factor : int
        Factor of image augmentation
    data_path : str
        Base path of temporary directories for training data
    in_channels : int
        Number of input channels of training data
    out_channels : int
        Number of output channels of training data
    dilate_mask
        Radius of binary dilation of masks [-2, -1, 0, 1, 2]
    dilate_kernel : str
        Dilation kernel ('disk' or 'square')
    add_tile : int
        Add additional tile for splitting images into tiles with more overlapping tiles
    val_split : float
        Validation split for training
    invert : bool
        If True, greyscale binary labels is inverted
    skeletonize : bool
        If True, binary labels are skeletonized
    clip_threshold : Tuple[float, float]
        Clip thresholds for intensity normalization of images
    shiftscalerotate : [float, float, float]
        Shift, scale and rotate image during augmentation
    noise_amp : float
        Amplitude of Gaussian noise for image augmentation
    brightness_contrast : Tuple[float, float]
        Adapt brightness and contrast of images during augmentation
    rescale : float, optional
        Rescale all images and labels by factor rescale
    create : bool, optional
        If False, existing data set in data_path is used
    """

    def __init__(self, source_dir, dim_out=(256, 256), aug_factor=10, data_path='../data/', in_channels=1,
                 out_channels=1, dilate_mask=0, dilate_kernel='disk', add_tile=0,
                 val_split=0.2, invert=False, skeletonize=False, clip_threshold=(0.2, 99.8), shiftscalerotate=(0, 0, 0),
                 noise_lims=(0.5, 1.2), brightness_contrast=(0.25, 0.25), blur_limit=(2, 7), rescale=None, create=True):

        self.source_dir = source_dir
        self.create = create
        self.data_path = data_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_out = dim_out
        self.skeletonize = skeletonize
        self.invert = invert
        self.clip_threshold = clip_threshold
        self.add_tile = add_tile
        self.aug_factor = aug_factor
        self.shiftscalerotate = shiftscalerotate
        self.brightness_contrast = brightness_contrast
        self.noise_lims = noise_lims
        self.dilate_mask = dilate_mask
        self.dilate_kernel = dilate_kernel
        self.blur_limit = blur_limit
        self.val_split = val_split
        self.mode = 'train'

        self.__make_dirs()
        if create:
            self.__move_and_edit()
            self.__merge_images()
            self.__split()
            if self.aug_factor is not None:
                self.__augment()

    def __make_dirs(self):
        self.image_path = self.data_path + '/image/'
        self.mask_path = self.data_path + '/mask/'
        self.merge_path = self.data_path + '/merge/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.split_image_path = self.data_path + '/split/image/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.aug_image_path = self.data_path + '/augmentation/aug_image/'
        self.aug_mask_path = self.data_path + '/augmentation/aug_mask/'

        # delete old files
        if self.create:
            try:
                shutil.rmtree(self.data_path)
            except:
                pass
        # make folders
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_image_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_image_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)

    def __move_and_edit(self):
        # create image data
        files_image = [f for f in glob.glob(self.source_dir[0] + '*')
                       if f.lower().endswith(('.tif', '.tiff')) and not os.path.basename(f).startswith('.')]
        for file_i in files_image:
            img_i = tifffile.imread(file_i).astype(np.float32)
            # clip and normalize (0,255)
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i, self.clip_threshold[0]),
                            a_max=np.percentile(img_i, self.clip_threshold[1]))
            img_i = (img_i - np.nanmin(img_i)) / (np.nanmax(img_i) - np.nanmin(img_i)) * 255
            img_i = img_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            if len(img_i.shape) == 2:
                img_i = np.expand_dims(img_i, 0)
            tifffile.imwrite(self.image_path + save_i + '.tif', img_i)
        # create masks
        files_mask = [f for f in glob.glob(self.source_dir[1] + '*')
                      if f.lower().endswith(('.tif', '.tiff')) and not os.path.basename(f).startswith('.')]
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            mask_i = tifffile.imread(file_i)
            if len(mask_i.shape) == 2:
                mask_i = np.expand_dims(mask_i, 0)
            for j, ch_j in enumerate(mask_i):  # iterate channels
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

    def __merge_images(self):
        self.mask_files = glob.glob(self.data_path + '/mask/*.tif')
        self.image_files = glob.glob(self.data_path + '/image/*.tif')

        if len(self.mask_files) != len(self.image_files):
            raise ValueError('Number of ground truth does not match number of image stacks')

        for i, file_i in enumerate(self.mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            img_i = tifffile.imread(self.data_path + '/image/' + basename_i)
            merge = np.zeros((mask_i.shape[1], mask_i.shape[2], mask_i.shape[0] + img_i.shape[0]))
            merge[:, :, :mask_i.shape[0]] = np.moveaxis(mask_i, 0, 2)
            merge[:, :, mask_i.shape[0]:] = np.moveaxis(img_i, 0, 2)
            merge = merge.astype('uint8')
            tifffile.imwrite(self.merge_path + str(i) + '.tif', merge)

    def __split(self):
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
            N_x += self.add_tile if N_x > 1 else 0
            N_y += self.add_tile if N_y > 1 else 0
            # starting indices of patches
            X_start = np.linspace(0, dim_in[0] - self.dim_out[0], N_x).astype('int16')
            Y_start = np.linspace(0, dim_in[1] - self.dim_out[1], N_y).astype('int16')
            for j in range(N_x):
                for k in range(N_y):
                    patch_ij = merge[X_start[j]:X_start[j] + self.dim_out[0], Y_start[k]:Y_start[k] + self.dim_out[1],
                               :]
                    mask_ij = patch_ij[:, :, :self.out_channels]
                    image_ij = patch_ij[:, :, self.out_channels:]
                    tifffile.imwrite(self.split_merge_path + f'{n}.tif', patch_ij)
                    tifffile.imwrite(self.split_mask_path + f'{n}.tif', mask_ij)
                    tifffile.imwrite(self.split_image_path + f'{n}.tif', image_ij)
                    n += 1

    def __augment(self, p=0.8):
        aug_pipeline = Compose(transforms=[
            Flip(),
            RandomRotate90(p=1.0),
            ShiftScaleRotate(self.shiftscalerotate[0], self.shiftscalerotate[1], self.shiftscalerotate[2]),
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=0.5),
            Blur(blur_limit=self.blur_limit, always_apply=False, p=0.2),
            MultiplicativeNoise(multiplier=self.noise_lims, elementwise=True, p=0.3),
        ],
            p=p)

        patches_image = glob.glob(self.split_image_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')
        n_patches = len(patches_image)
        k = 0
        for i in range(n_patches):
            image_i = tifffile.imread(patches_image[i])
            mask_i = tifffile.imread(patches_mask[i])

            data_i = {'image': image_i, 'mask': mask_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            imgs_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])

            for j in range(self.aug_factor):
                tifffile.imwrite(self.aug_image_path + f'{k}.tif', np.moveaxis(imgs_aug_i[j], 2, 0))
                tifffile.imwrite(self.aug_mask_path + f'{k}.tif', np.moveaxis(masks_aug_i[j], 2, 0))
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
        else:
            image_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
        image = torch.from_numpy(image_0)
        mask = torch.from_numpy(mask_0)
        del image_0, mask_0
        sample = {'image': image, 'mask': mask}
        return sample
