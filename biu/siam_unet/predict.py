import os, glob, re, shutil
import subprocess
from sys import platform

import tifffile
import numpy as np
from tifffile.tifffile import TiffFile
from tqdm import tqdm as tqdm

import torch
from torch import nn as nn, flatten

from .helpers.util import write_info_file
from .helpers.__md5sum__ import md5sum, md5sum_folder
import cv2
from .siam_unet import Siam_UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Predict:
    """
    Class for prediction of tif-movies.
    1) Loading file and preprocess (normalization)
    2) Resizing of images into patches with resize_dim
    3) Prediction with U-Net
    4) Stitching of predicted patches and averaging of overlapping regions
    """

    def __init__(self, tif_file, result_name, model_params, n_filter=64, resize_dim=(512, 512), invert=False, clip_thres=(0.2, 99.8), add_tile=0, bias=1):
        """Predicts a tif movie

        Args:
            tif_file (str): Path to input tif stack
            result_name (str): 
            model_params (str): Path to model parameters file. Usually ends in `.pth`.
            n_filter (int, optional): Number of filters used when training the model. Must use the same n_filter parameter for training and predicting. Defaults to 64.
            resize_dim (tuple, optional): Resize dimension. Same resize dimension must be used for training and predicting. Defaults to (512, 512).
            invert (bool, optional): Whether to invert the output. Defaults to False.
            clip_thres (tuple, optional): Threshold over which to clip the mask and image pair. Defaults to (0.2, 99.8).
            add_tile (int, optional): How many additional patches to add to each axis. Changing this parameter should be a miniscule difference. Defaults to 0.
            bias (int, optional): Passed to the `bias` of Siam-UNet when predicting. Would add a bias term to the network output before its final sigmoid gate. Defaults to 1.
        """
        self.tif_file = tif_file
        self.resize_dim = resize_dim
        self.add_tile = add_tile
        self.n_filter = n_filter
        self.bias = bias
        self.invert = invert
        self.clip_thres = clip_thres
        self.result_name = result_name
        if self.result_name == 'nodes':
            self.folder = os.path.dirname(self.tif_file)
        else:
            self.folder = re.split('.tif', self.tif_file)[0] + '/'

        write_info_file(result_name + '.info.txt', f'Mode: Predict with Siam_UNet\nOutfile name: {result_name}\nInput file: {tif_file}\nHash: {md5sum(tif_file)}Model: {model_params}\nModel hash:{md5sum(model_params)}')

        # load model
        self.model = Siam_UNet(n_filter=self.n_filter, bias=self.bias).to(device)
        self.model.load_state_dict(torch.load(model_params)['state_dict'])
        self.model.eval()

        # split data into groups of two images
        tif_key = TiffFile(self.tif_file)
        self.tif_len = len(tif_key.pages)
        self.imgs_shape = [self.tif_len, tif_key.pages[0].shape[0], tif_key.pages[0].shape[1]]

        temp_dir = f'temp_{self.tif_file.split("/")[-1]}'
        # os.system(f"mkdir -p \'{temp_dir}\'")

        # taken from split()
        # number of patches in x and y
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0])) + self.add_tile
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1])) + self.add_tile
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_x * self.N_y   # total number of patches

        os.makedirs(temp_dir, exist_ok=True)
        print('Predicting data ...')
        for i in tqdm(range(self.tif_len), unit='frame'):
            if i == 0:
                prev_img = tifffile.imread(self.tif_file, key=1)
            else:
                prev_img = current_img
            current_img = tifffile.imread(self.tif_file, key=i)
            
            img_stack = np.array([prev_img, current_img])
            img_stack = self.preprocess(img_stack)
            patches = self.split(img_stack)
            _ = print(f'Patches shape:{patches.shape}') if i == 0 else None
            result_patches = self.predict(patches)
            imgs_result = self.stitch(result_patches)
            cv2.imwrite(filename=f'{temp_dir}/{i}.tif', img=imgs_result.astype('uint8'), )


        # predict each pair, and save the output of each one as a separate image

        # merge the images and save as tif file
        print(f'Saving prediction results as {result_name}...')
        tifffile.imwrite(data=tqdm(self.individual_tif_generator(dir=temp_dir), total=self.tif_len, unit='frame'), file=self.result_name, dtype=np.uint8, shape=self.imgs_shape)
        os.system(f'rm -rf \'{temp_dir}\'')

    def individual_tif_generator(self, dir):
        # a generator that returns each frame in the directory
        for i in range(self.tif_len):
            yield tifffile.imread(f'{dir}/{i}.tif')

    def open_folder(self):
        if platform.system() == "Windows":
            os.startfile(self.folder)
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", self.folder])


    def preprocess(self, imgs):
        if len(imgs.shape) == 3:
            for i, img in enumerate(imgs):
                img = np.clip(img, a_min=np.nanpercentile(img, self.clip_thres[0]),
                              a_max=np.percentile(img, self.clip_thres[1]))
                img = img - np.min(img)
                img = img / np.max(img) * 255
                if self.invert:
                    img = 255 - img
                imgs[i] = img
        if len(imgs.shape) == 2:
            imgs = np.clip(imgs, a_min=np.nanpercentile(imgs, self.clip_thres[0]),
                           a_max=np.percentile(imgs, self.clip_thres[1]))
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs) * 255
            if self.invert:
                imgs = 255 - imgs
        imgs = imgs.astype('uint8')
        return imgs

    def split(self, imgs):
        # define array for prediction
        patches = np.zeros((self.N, 2, self.resize_dim[0], self.resize_dim[1]), dtype='uint8')

        # zero padding of image if imgs_shape < resize_dim
        if self.imgs_shape[0] > 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)),
                              'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                imgs = np.pad(imgs, ((0, 0), (0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])),
                              'constant')
        elif self.imgs_shape[0] == 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                imgs = np.pad(imgs, ((0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)), 'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])), 'constant')

        # starting indices of patches
        self.X_start = np.linspace(0, self.imgs_shape[1] - self.resize_dim[0], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, self.imgs_shape[2] - self.resize_dim[1], self.N_y).astype('uint16')

        # split in resize_dim
        n = 0
        if self.imgs_shape[0] > 1: # If our input image has more than one frame
            i = 1
            for j in range(self.N_x):
                for k in range(self.N_y):
                    patches[n, 0, :, :] = imgs[i][self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                            self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    patches[n, 1, :, :] = imgs[i - 1][self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                        self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    n += 1
        elif self.imgs_shape[0] == 1:
            raise Exception # I didn't implement this part
            for j in range(self.N_x):
                for k in range(self.N_y):
                    patches[n, 0, :, :] = imgs[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                          self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    n += 1
        return patches

    def predict(self, patches):
        result_patches = np.zeros((patches.shape[0], 1, patches.shape[2], patches.shape[3]), dtype='uint8')
        with torch.no_grad():
            for i, patch_i in enumerate(patches):
                image_patch_i = patch_i[0, :, :]
                prev_image_patch_i = patch_i[1, :, :]

                image_patch_i = torch.from_numpy(image_patch_i.astype('float32') / 255).to(device).view((1, 1, self.resize_dim[0], self.resize_dim[1]))
                prev_image_patch_i = torch.from_numpy(prev_image_patch_i.astype('float32') / 255).to(device).view((1, 1, self.resize_dim[0], self.resize_dim[1]))

                res_i = self.model(image_patch_i, prev_image_patch_i).view((1, self.resize_dim[0], self.resize_dim[1])).cpu().numpy() * 255
                result_patches[i] = res_i.astype('uint8')
                del patch_i, res_i
        return result_patches

    def stitch(self, result_patches):
        # create array
        imgs_result = np.zeros((self.imgs_shape[0], np.max((self.resize_dim[0], self.imgs_shape[1]))
                                , np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8')
        i = 0
        if self.imgs_shape[0] > 1:  # if stack
            stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                        np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8') * np.nan
        elif self.imgs_shape[0] == 1:  # if only one image
            stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                        np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8') * np.nan
        n = 0
        for j in range(self.N_x):
            for k in range(self.N_y):
                stack_result_i[n, self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]] = result_patches[i * self.N_per_img + n, 0, :, :]
                n += 1
        # average overlapping regions
        imgs_result = np.nanmean(stack_result_i, axis=0)
        del stack_result_i

        # change to input size (if zero padding)
        imgs_result = imgs_result[:self.imgs_shape[1], :self.imgs_shape[2]]

        return imgs_result

    def save_as_tif(self, imgs, filename):
        imgs = imgs.astype('uint8')
        tifffile.imsave(filename, imgs)
        print('Saving prediction results as %s' % filename)

