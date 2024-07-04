from typing import Union

import numpy as np
import tifffile
import torch

from ..progress import ProgressNotifier
from .unet import Unet
from .unet_v0 import Unet_v0
from .attention_unet import AttentionUnet
from ..utils import save_as_tif, get_device


class Predict:
    """
    Class for prediction of movies and images with U-Net

    1) Loading file and preprocess (normalization)
    2) Resizing of images into patches with resize_dim
    3) Prediction with U-Net
    4) Stitching of predicted patches and averaging of overlapping regions

    Parameters
    ----------
    imgs : ndarray, string
        images to predict, if string, attempt to load from tif file
    result_name : str
        path for result
    model_params : str
        path of u-net parameters (.pth file)
    network
        Network class
    resize_dim
        Image dimensions for resizing for prediction
    invert : bool
        Invert greyscale of image(s) before prediction
    normalization_mode : str
        Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
        'all': based on histogram of full stack, 'first': based on histogram of first image in stack)
    clip_threshold : Tuple[float, float]
        Clip threshold for image intensity before prediction
    add_tile : int, optional
        Add additional tiles for splitting large images to increase overlap
    normalize_result : bool
        If true, results are normalized to [0, 255]
    show_progress : bool
        Whether to show progress bar.
    device : torch.device or str, optional
        Device to run the pytorch model on, defaults to 'auto', which selects CUDA or MPS if available.
    progress_notifier:
        Wrapper to show tqdm progress notifier in gui
    """

    def __init__(self, imgs, result_name, model_params, network='Unet', resize_dim=(512, 512),
                 invert=False, normalization_mode='single', clip_threshold=(0., 99.8), add_tile=0,
                 normalize_result=False, show_progress=True, device: Union[torch.device, str] = 'auto',
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):

        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        if isinstance(imgs, str):
            imgs = tifffile.imread(imgs)

        self.resize_dim = resize_dim
        self.add_tile = add_tile
        self.normalize_result = normalize_result
        self.invert = invert
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.result_name = result_name
        self.show_progress = show_progress

        # read, preprocess and split data
        imgs = self.__reshape_data(imgs)
        imgs = self.__preprocess(imgs)
        patches = self.__split(imgs)
        del imgs

        # load model and predict data
        self.model_params = torch.load(model_params, map_location=self.device)
        if network is None:
            if 'network' in self.model_params.keys():
                network = self.model_params['network']
            else:
                raise ValueError('network is not defined')
        if network == 'Unet':
            network = Unet
        elif network == 'AttentionUnet':
            network = AttentionUnet
        elif network == 'Unet_v0':
            network = Unet_v0
            if 'in_channels' not in self.model_params.keys():
                self.model_params['in_channels'] = 1
                self.model_params['out_channels'] = 1
        self.model = network(n_filter=self.model_params['n_filter'], in_channels=self.model_params['in_channels'],
                             out_channels=self.model_params['out_channels']).to(self.device)
        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()
        result_patches = self.__predict(patches, progress_notifier)
        del patches, self.model

        # stitch patches (mean of overlapped regions)
        imgs_result = self.__stitch(result_patches)
        del result_patches, self.model_params

        # save as .tif file
        save_as_tif(imgs_result, self.result_name, normalize=normalize_result)
        del imgs_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __reshape_data(self, imgs):
        self.imgs_shape = imgs.shape
        if len(self.imgs_shape) == 2:  # if single image
            imgs = np.expand_dims(imgs, axis=0)
            self.imgs_shape = imgs.shape
        return imgs

    def __preprocess(self, imgs):
        if self.normalization_mode == 'single':
            for i, img in enumerate(imgs):
                img = np.clip(img, a_min=np.nanpercentile(img, self.clip_threshold[0]),
                              a_max=np.percentile(img, self.clip_threshold[1]))
                img = img - np.min(img)
                img = img / np.max(img) * 255
                if self.invert:
                    img = 255 - img
                imgs[i] = img
        elif self.normalization_mode == 'first':
            clip_threshold = (np.nanpercentile(imgs[0], self.clip_threshold[0]),
                              np.percentile(imgs[0], self.clip_threshold[1]))
            imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs) * 255
            if self.invert:
                imgs = 255 - imgs
        elif self.normalization_mode == 'all':
            clip_threshold = (np.nanpercentile(imgs, self.clip_threshold[0]),
                              np.percentile(imgs, self.clip_threshold[1]))
            imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs) * 255
            if self.invert:
                imgs = 255 - imgs
        else:
            raise ValueError(f'normalization_mode {self.normalization_mode} not valid!')
        return imgs

    def __split(self, imgs):
        # number of patches in x and y
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0])) + self.add_tile
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1])) + self.add_tile
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_x * self.N_y * self.imgs_shape[0]  # total number of patches

        # define array for prediction
        patches = np.zeros((self.N, 1, self.resize_dim[0], self.resize_dim[1]), dtype='uint8')

        # zero padding of image if imgs_shape < resize_dim
        if self.resize_dim[0] > self.imgs_shape[1]:  # for x
            imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)),
                          'reflect')
        if self.resize_dim[1] > self.imgs_shape[2]:  # for y
            imgs = np.pad(imgs, ((0, 0), (0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])),
                          'reflect')

        # starting indices of patches
        self.X_start = np.linspace(0, self.imgs_shape[1] - self.resize_dim[0], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, self.imgs_shape[2] - self.resize_dim[1], self.N_y).astype('uint16')

        # split in resize_dim
        n = 0
        for i, img in enumerate(imgs):
            for j in range(self.N_x):
                for k in range(self.N_y):
                    patches[n, 0, :, :] = img[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                              self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    n += 1
        return patches

    def __predict(self, patches, progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        result_patches = np.zeros((patches.shape[0], self.model_params['out_channels'], *patches.shape[2:]),
                                  dtype='uint8')
        print('Predicting data ...') if self.show_progress else None
        with torch.no_grad():
            _progress_notifier = enumerate(progress_notifier.iterator(patches)) if self.show_progress else enumerate(
                patches)
            for i, patch_i in _progress_notifier:
                patch_i = torch.from_numpy(patch_i.astype('float32') / 255).to(self.device).view((1,
                                                                                             self.model_params[
                                                                                                 'in_channels'],
                                                                                             self.resize_dim[0],
                                                                                             self.resize_dim[1]))
                res_i, logits_i = self.model(patch_i)
                res_i = res_i.view(
                    (self.model_params['out_channels'], self.resize_dim[0], self.resize_dim[1])).cpu().numpy()
                result_patches[i] = (res_i * 255).astype('uint8')
                del patch_i, res_i
        return result_patches

    def __stitch(self, result_patches):
        # create array
        imgs_result = np.zeros(
            (self.imgs_shape[0], self.model_params['out_channels'], np.max((self.resize_dim[0], self.imgs_shape[1])),
             np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8')
        for i in range(self.imgs_shape[0]):

            stack_result_i = np.zeros((self.N_per_img, self.model_params['out_channels'],
                                       np.max((self.resize_dim[0], self.imgs_shape[1])),
                                       np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8') * np.nan

            n = 0
            for j in range(self.N_x):
                for k in range(self.N_y):
                    stack_result_i[n, :, self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                    self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]] = result_patches[i * self.N_per_img + n]
                    n += 1

            # average overlapping regions
            imgs_result[i] = np.nanmean(stack_result_i, axis=0)
            del stack_result_i

        # change to input size (if zero padding)
        imgs_result = imgs_result[:, :, :self.imgs_shape[1], :self.imgs_shape[2]]

        return np.squeeze(imgs_result)
