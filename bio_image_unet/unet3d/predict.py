from typing import Union

import numpy as np
import tifffile
import torch

from .unet3d import UNet3D
from ..progress import ProgressNotifier
from ..utils import save_as_tif, get_device


class Predict:
    """
    Class for prediction of movies or 3D stacks with 3D U-Net

    1) Loading file and preprocess (normalization)
    2) Resizing of images into patches with resize_dim
    3) Prediction with U-Net
    4) Stitching of predicted patches and averaging of overlapping regions

    Parameters
    ----------
    vol : ndarray, string
        volume to predict, if string, attempt to load from tif file
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
    add_patch : int, optional
        Add additional patches for splitting large images to increase overlap
    normalize_result : bool
        If true, results are normalized to [0, 255]
    progress_bar : bool, optional
        Whether to display progress bar during prediction.
    device : torch.device, optional
        Device to run pytorch model on. Default is 'auto', which selects CUDA or MPS if available.
    progress_notifier:
        Wrapper to show tqdm progress notifier in gui
    """

    def __init__(self, vol, result_name, model_params, network=UNet3D, resize_dim=(512, 512),
                 invert=False, normalization_mode='single', clip_threshold=(0., 99.8), add_patch=0,
                 normalize_result=False, progress_bar=True, device: Union[torch.device, str] = 'auto',
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):

        if isinstance(vol, str):
            vol = tifffile.imread(vol)

        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.resize_dim = resize_dim
        self.add_patch = add_patch
        self.normalize_result = normalize_result
        self.invert = invert
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.result_name = result_name
        self.progress_bar = progress_bar

        # read, preprocess and split data
        vol = self.__reshape_data(vol)
        vol = self.__preprocess(vol)
        patches = self.__split(vol)
        del vol

        # load model and predict data
        self.model_params = torch.load(model_params, map_location=self.device)
        if 'use_interpolation' not in self.model_params.keys():
            self.model_params['use_interpolation'] = False
        self.model = network(n_filter=self.model_params['n_filter'], in_channels=self.model_params['in_channels'],
                             out_channels=self.model_params['out_channels'],
                             use_interpolation=self.model_params['use_interpolation']).to(self.device)
        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()
        result_patches = self.__predict(patches, progress_notifier)
        del patches, self.model

        # stitch patches (mean of overlapped regions)
        vol_result = self.__stitch(result_patches)
        del result_patches, self.model_params

        # save as .tif file
        save_as_tif(vol_result, self.result_name, normalize=normalize_result)
        del vol_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __reshape_data(self, vol):
        self.vol_shape = vol.shape
        if len(self.vol_shape) == 2:  # if single image
            vol = np.expand_dims(vol, axis=0)
            self.vol_shape = vol.shape
        return vol

    def __preprocess(self, vol):
        clip_threshold = (np.nanpercentile(vol, self.clip_threshold[0]),
                          np.percentile(vol, self.clip_threshold[1]))
        vol = np.clip(vol, clip_threshold[0], clip_threshold[1])
        vol = vol - np.min(vol)
        vol = vol / np.max(vol) * 255
        if self.invert:
            vol = 255 - vol
        return vol

    def __split(self, vol):
        # number of patches in x and y
        self.N_z = int(np.ceil(self.vol_shape[0] / self.resize_dim[0])) + self.add_patch
        self.N_x = int(np.ceil(self.vol_shape[1] / self.resize_dim[1])) + self.add_patch
        self.N_y = int(np.ceil(self.vol_shape[2] / self.resize_dim[2])) + self.add_patch
        self.N_x += self.add_patch if self.N_z > 1 else 0
        self.N_x += self.add_patch if self.N_x > 1 else 0
        self.N_y += self.add_patch if self.N_y > 1 else 0

        self.N = self.N_x * self.N_y * self.N_z  # total number of patches

        # define array for prediction
        patches = np.zeros((self.N, self.resize_dim[0], self.resize_dim[1], self.resize_dim[2]), dtype='uint8')

        # zero padding of image if vol_shape < resize_dim
        z_gap = max(0, self.resize_dim[0] - self.vol_shape[0])
        x_gap = max(0, self.resize_dim[1] - self.vol_shape[1])
        y_gap = max(0, self.resize_dim[2] - self.vol_shape[2])
        vol = np.pad(vol, ((0, z_gap), (0, x_gap), (0, y_gap)), 'reflect')

        # starting indices of patches
        self.Z_start = np.linspace(0, self.vol_shape[0] - self.resize_dim[0], self.N_z).astype('uint16')
        self.X_start = np.linspace(0, self.vol_shape[1] - self.resize_dim[1], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, self.vol_shape[2] - self.resize_dim[2], self.N_y).astype('uint16')

        # split in resize_dim
        n = 0
        for j in range(self.N_z):
            for k in range(self.N_x):
                for p in range(self.N_y):
                    patches[n, :, :, :] = vol[self.Z_start[j]:self.Z_start[j] + self.resize_dim[0],
                                          self.X_start[k]:self.X_start[k] + self.resize_dim[1],
                                          self.Y_start[p]:self.Y_start[p] + self.resize_dim[2]]
                    n += 1
        return patches

    def __predict(self, patches, progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        result_patches = np.zeros_like(patches, dtype='uint8')
        print('Predicting data ...') if self.progress_bar else None
        with torch.no_grad():
            _progress_notifier = enumerate(progress_notifier.iterator(patches)) if self.progress_bar else enumerate(
                patches)
            for i, patch_i in _progress_notifier:
                patch_i = torch.from_numpy(patch_i.astype('float32') / 255).to(self.device).view((1, 1,
                                                                                                  self.resize_dim[0],
                                                                                                  self.resize_dim[1],
                                                                                                  self.resize_dim[2]))
                res_i, logits_i = self.model(patch_i)
                res_i = res_i.view(
                    (self.resize_dim[0], self.resize_dim[1], self.resize_dim[2])).cpu().numpy()
                result_patches[i] = (res_i * 255).astype('uint8')
                del patch_i, res_i
        return result_patches

    def __stitch(self, result_patches):
        # create array
        _vol_result = np.zeros(shape=(3, max(self.vol_shape[0], self.resize_dim[0]),
                                      max(self.resize_dim[1], self.vol_shape[1]),
                                      max(self.resize_dim[2], self.vol_shape[2])), dtype='float16') * np.nan

        n = 0
        for i in range(self.N_z):
            for j in range(self.N_x):
                for k in range(self.N_y):
                    _vol_result[np.mod(n, 3),
                    self.Z_start[i]:self.Z_start[i] + self.resize_dim[0],
                    self.X_start[j]:self.X_start[j] + self.resize_dim[1],
                    self.Y_start[k]:self.Y_start[k] + self.resize_dim[2]] = result_patches[n]
                    n += 1

        # average overlapping regions
        vol_result = np.nanmean(_vol_result, axis=0).astype('uint8')

        # change to input size (if zero padding)
        vol_result = vol_result[:self.vol_shape[0], :self.vol_shape[1], :self.vol_shape[2]]

        return np.squeeze(vol_result)
