import os.path
from typing import Union

import numpy as np
import tifffile
import torch

from ..progress import ProgressNotifier
from .multi_output_nested_unet import MultiOutputNestedUNet, MultiOutputNestedUNet_3Levels
from ..utils import get_device


class Predict:
    """Class for prediction of movies and images using a U-Net model."""

    def __init__(self, imgs, model_params, result_path=None, network=MultiOutputNestedUNet, resize_dim=(1024, 1024),
                 batch_size=1, normalization_mode='single', clip_threshold=(0., 99.98), add_tile=0,
                 show_progress=True, device: Union[torch.device, str] = 'auto',
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Initialize the Predict class for performing predictions on TIFF files using a standard 2D U-Net.

        The process involves:
        1) Loading and preprocessing the input images (including normalization).
        2) Resizing images into patches based on the specified dimensions.
        3) Performing predictions using the U-Net model.
        4) Stitching predicted patches together and averaging overlapping regions.

        Parameters
        ----------
        imgs : ndarray or str
            Images to predict. If a string is provided, it attempts to load images from a TIFF file.
        model_params : str
            Path to the U-Net model parameters (.pt file).
        result_path : str, optional
            Path to save the prediction results as a TIFF file. If None, results are stored in the 'result' attribute.
        network : Network, optional
            Network architecture. Default is MultiOutputUnet
        resize_dim : Tuple[int, int]
            Dimensions to resize images for prediction.
        batch_size : int
            Number of images to process in each batch during prediction.
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction. Options are:
            - 'single': Normalize each image individually.
            - 'all': Normalize based on the histogram of the entire stack.
            - 'first': Normalize based on the histogram of the first image in the stack.
        clip_threshold : Tuple[float, float]
            Thresholds for clipping image intensity before prediction.
        add_tile : int, optional
            Additional tiles to add for splitting large images to increase overlap.
        show_progress : bool
            Whether to display a progress bar during prediction.
        device : torch.device or str, optional
            Device to run the PyTorch model on. Defaults to 'auto', which selects CUDA or MPS if available.
        progress_notifier : ProgressNotifier, optional
            Wrapper to display a progress notifier in the GUI using tqdm.
        """
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        if isinstance(imgs, str):
            imgs = tifffile.imread(imgs)

        self.resize_dim = resize_dim
        self.batch_size = batch_size
        self.add_tile = add_tile
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.result_path = result_path
        self.show_progress = show_progress

        # read, preprocess and split data
        imgs = imgs.astype('float32')
        imgs = self.__reshape_data(imgs)
        imgs = self.__preprocess(imgs)
        patches = self.__split(imgs)
        del imgs

        # load model and predict data
        self.model_params = torch.load(model_params, map_location=self.device)
        self.model = network(in_channels=self.model_params['in_channels'],
                             n_filter=self.model_params['n_filter'],
                             output_heads=self.model_params['output_heads'],
                             deep_supervision=self.model_params.get('deep_supervision', False)).to(self.device)
        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()
        self.target_keys = self.model_params['output_heads'].keys()
        result_patches = self.__predict(patches, self.batch_size, progress_notifier)
        del patches, self.model

        # stitch patches (mean of overlapped regions)
        result = self.__stitch(result_patches)
        del result_patches, self.model_params

        if self.result_path is not None:
            # save as .tif files
            for target_key in self.target_keys:
                if os.path.isdir(self.result_path):
                    tifffile.imwrite(self.result_path + target_key + '.tif', result[target_key])
                else:
                    tifffile.imwrite(self.result_path + '_' + target_key + '.tif', result[target_key])
            del result
            self.result = None
        else:
            self.result = result
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
                img = img / np.max(img)
                imgs[i] = img
        elif self.normalization_mode == 'first':
            clip_threshold = (np.nanpercentile(imgs[0], self.clip_threshold[0]),
                              np.percentile(imgs[0], self.clip_threshold[1]))
            imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs)
        elif self.normalization_mode == 'all':
            clip_threshold = (np.nanpercentile(imgs, self.clip_threshold[0]),
                              np.percentile(imgs, self.clip_threshold[1]))
            imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs)
        else:
            raise ValueError(f'normalization_mode {self.normalization_mode} not valid!')
        return imgs

    def __split(self, imgs):
        # Calculate number of patches
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0])) + self.add_tile
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1])) + self.add_tile
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_per_img * self.imgs_shape[0]

        # Pad images if needed
        pad_x = max(self.resize_dim[0] - self.imgs_shape[1], 0)
        pad_y = max(self.resize_dim[1] - self.imgs_shape[2], 0)
        imgs = np.pad(imgs, ((0, 0), (0, pad_x), (0, pad_y)), 'reflect')

        # Calculate starting indices (preserved for stitching)
        self.X_start = np.linspace(0, imgs.shape[1] - self.resize_dim[0], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, imgs.shape[2] - self.resize_dim[1], self.N_y).astype('uint16')

        # Generate patches using vectorized operations
        window_shape = (self.resize_dim[0], self.resize_dim[1])
        patches = np.lib.stride_tricks.sliding_window_view(imgs, window_shape, axis=(1, 2))
        patches = patches[:, ::self.X_start[1] if self.N_x > 1 else 1,
                  ::self.Y_start[1] if self.N_y > 1 else 1]
        patches = patches.reshape(-1, *window_shape)

        return patches

    def __predict(self, patches, batch_size=16, progress_notifier=None):
        # Initialize result_patches with correct dimensions
        result_patches = {
            target_key: np.zeros(
                (patches.shape[0], self.model_params['output_heads'][target_key]['channels'], *self.resize_dim),
                dtype='float32'
            )
            for target_key in self.target_keys
        }

        print('Predicting data ...') if self.show_progress else None

        with torch.no_grad():
            num_batches = int(np.ceil(patches.shape[0] / batch_size))
            _progress_notifier = range(num_batches)
            if self.show_progress and progress_notifier:
                _progress_notifier = progress_notifier.iterator(_progress_notifier)

            for batch_idx in _progress_notifier:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, patches.shape[0])

                # Prepare batch patches
                batch_patches = patches[start_idx:end_idx]
                batch_patches = torch.tensor(batch_patches).to(self.device)
                batch_patches = batch_patches.view(
                    -1, self.model_params['in_channels'], self.resize_dim[0], self.resize_dim[1]
                )

                # Get model predictions
                batch_results = self.model(batch_patches)

                # Assign predictions to result_patches
                for target_key in self.target_keys:
                    # Extract model output for this target key
                    batch_results_target = batch_results[target_key].cpu().numpy()

                    # Ensure shapes match before assignment
                    expected_shape = result_patches[target_key][start_idx:end_idx].shape
                    if batch_results_target.shape != expected_shape:
                        raise RuntimeError(
                            f"Shape mismatch for target key '{target_key}': "
                            f"predicted {batch_results_target.shape}, expected {expected_shape}"
                        )

                    result_patches[target_key][start_idx:end_idx] = batch_results_target

        return result_patches

    def __stitch(self, result_patches):
        result = {}

        for target_key in self.target_keys:
            result_patches_target = result_patches[target_key]
            result_target = np.zeros(
                shape=(self.imgs_shape[0], self.model_params['output_heads'][target_key]['channels'],
                       max(self.resize_dim[0], self.imgs_shape[1]),
                       max(self.resize_dim[1], self.imgs_shape[2])),
                dtype='float32'
            )
            weight = np.zeros_like(result_target)

            for i in range(self.imgs_shape[0]):
                stack_result_i = result_patches_target[
                                 i * self.N_per_img:(i + 1) * self.N_per_img
                                 ].reshape(self.N_x, self.N_y, *result_patches_target.shape[1:])

                for j, x_start in enumerate(self.X_start):
                    for k, y_start in enumerate(self.Y_start):
                        x_slice = slice(x_start, x_start + self.resize_dim[0])
                        y_slice = slice(y_start, y_start + self.resize_dim[1])

                        # Add patch to result and update weight matrix
                        result_target[i, :, x_slice, y_slice] += stack_result_i[j, k]
                        weight[i, :, x_slice, y_slice] += 1

            # Normalize overlapping regions
            np.divide(result_target, weight, out=result_target, where=weight > 0)

            # Crop to original image size and remove unnecessary dimensions
            result_target = result_target[:, :, :self.imgs_shape[1], :self.imgs_shape[2]]
            result[target_key] = np.squeeze(result_target)

        return result
