import os.path
from typing import Union

import numpy as np
import tifffile
import torch

from ..progress import ProgressNotifier
from .multi_output_nested_unet import MultiOutputNestedUNet, MultiOutputNestedUNet_3Levels
from ..utils import get_device
from ..utils.model_cache import load_cached_model
from ..utils.tiling import TileBlender, auto_batch_size, auto_patch_size, pad_to_patch, plan_grid

# Ramp width at a seam, in pixels. Tiles are planned with at least this much
# overlap so the ramps have room; a narrower overlap just gets a shorter ramp.
BLEND_MARGIN = 24


class Predict:
    """Class for prediction of movies and images using a U-Net model."""

    def __init__(self, imgs, model_params, result_path=None, network=MultiOutputNestedUNet, max_patch_size='auto',
                 batch_size='auto', normalization_mode='single', clip_threshold=(0., 99.98), add_tile=0,
                 compress_tif=False, show_progress=True, device: Union[torch.device, str] = 'auto',
                 clip_values=None, cache_model=True, prune_level=None,
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
        max_patch_size : Tuple[int, int] or 'auto'
            Maximal patch dimensions. 'auto' derives them from free device memory
            and the model's filter count, preferring a single patch over the whole
            image when it fits -- fewer tiles means less redundant compute. An
            explicit tuple is an upper bound: the patch is shrunk below it where
            that lets a whole number of tiles cover the image with less overlap.
        batch_size : int or 'auto'
            Number of patches per forward pass. 'auto' sizes the batch from free
            GPU memory on CUDA and uses 1 elsewhere, where batching does not help.
            It costs memory on top of the patch, not instead of it, so an
            auto-sized patch resolves to a batch of 1; small patches leave room
            for more.
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction. Options are:
            - 'single': Normalize each image individually.
            - 'all': Normalize based on the histogram of the entire stack.
            - 'first': Normalize based on the histogram of the first image in the stack.
        clip_threshold : Tuple[float, float]
            Thresholds for clipping image intensity before prediction.
        add_tile : int, optional
            Additional tiles to add for splitting large images to increase overlap.
        compress_tif : bool, optional
            Whether to compress tif files to save storage. Default is False.
        show_progress : bool
            Whether to display a progress bar during prediction.
        device : torch.device or str, optional
            Device to run the PyTorch model on. Defaults to 'auto', which selects CUDA or MPS if available.
        clip_values : Tuple[float, float], optional
            Absolute intensity limits to clip to, bypassing the percentile pass over
            ``clip_threshold``. Lets a caller predict a stack block by block while
            keeping normalization identical to a whole-stack run. Default is None.
        cache_model : bool, optional
            Keep the loaded weights in a process-level cache so repeated calls with
            the same checkpoint skip the load. Default is True.
        prune_level : int, optional
            For nested (UNet++) networks trained with deep supervision, stop at this
            nesting depth and read the matching output head. Level 2 costs about
            half the operations of the full model; how much accuracy that gives up
            depends on the checkpoint, so measure before using it. None keeps the
            network's own default (the deepest head). Default is None.
        progress_notifier : ProgressNotifier, optional
            Wrapper to display a progress notifier in the GUI using tqdm.
        """
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.precision = torch.float16 if self.device.type == 'cuda' or self.device.type == 'mps' else torch.float32

        if isinstance(imgs, str):
            imgs = tifffile.imread(imgs)

        self.max_patch_size = max_patch_size
        self.add_tile = add_tile
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.clip_values = clip_values
        self.result_path = result_path
        self.compress_tif = compress_tif
        self.show_progress = show_progress

        # read and preprocess data
        imgs = imgs.astype('float32')
        imgs = self.__reshape_data(imgs)
        imgs = self.__preprocess(imgs)

        # load the model first: its filter count sets the memory cost per pixel,
        # which is what an 'auto' patch size is derived from
        self.model, self.model_params = load_cached_model(
            model_params, network, self.device, self.precision, cache_model,
            from_params={'deep_supervision': False}, train_mode=False,
            **({} if prune_level is None else {'prune_level': prune_level}))
        self.target_keys = list(self.model_params['output_heads'].keys())

        result_patches = self.__split_and_predict(imgs, batch_size, progress_notifier)
        del imgs
        self.model = None

        # stitch patches (weighted mean of overlapped regions)
        result = self.__stitch(result_patches)
        del result_patches

        if self.result_path is not None:
            # save as .tif files
            for target_key in self.target_keys:
                target_file = self.result_path + target_key + '.tif' if os.path.exists(self.result_path) else (
                        self.result_path + '_' + target_key + '.tif')
                if self.compress_tif:
                    tifffile.imwrite(target_file, result[target_key], compression='deflate',
                                     compressionargs={'level': 6}, predictor=3)
                else:
                    tifffile.imwrite(target_file, result[target_key],)
            del result
            self.result = None
        else:
            self.result = result
        del self.model_params
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __reshape_data(self, imgs):
        self.imgs_shape = imgs.shape
        if len(self.imgs_shape) == 2:  # if single image
            imgs = np.expand_dims(imgs, axis=0)
            self.imgs_shape = imgs.shape
        return imgs

    def __preprocess(self, imgs):
        if self.clip_values is not None:
            lo, hi = self.clip_values
            np.clip(imgs, lo, hi, out=imgs)
            imgs -= lo
            if hi > lo:
                imgs /= (hi - lo)
            return imgs
        if self.normalization_mode == 'single':
            for i, img in enumerate(imgs):
                img = np.clip(img, a_min=np.nanpercentile(img, self.clip_threshold[0]),
                              a_max=np.percentile(img, self.clip_threshold[1]))
                img = img - np.min(img)
                img = img / np.max(img)
                imgs[i] = img
        elif self.normalization_mode in ('first', 'all'):
            reference = imgs[0] if self.normalization_mode == 'first' else imgs
            lo, hi = (np.nanpercentile(reference, self.clip_threshold[0]),
                      np.percentile(reference, self.clip_threshold[1]))
            np.clip(imgs, lo, hi, out=imgs)
            imgs -= imgs.min()
            peak = imgs.max()
            if peak > 0:
                imgs /= peak
        else:
            raise ValueError(f'normalization_mode {self.normalization_mode} not valid!')
        return imgs

    def __split_and_predict(self, imgs, batch_size, progress_notifier):
        """Tile, run the model, and back off if the device runs out of memory.

        The batch gives way first, leaving the tile grid and so the result alone;
        only then is the patch halved. Pinning the batch at 1 from the first
        failure is what keeps each attempt smaller than the last, since the batch
        is derived from the cost of a patch.
        """
        cap = self.max_patch_size
        batch_cap = None
        for attempt in range(4):
            if isinstance(cap, str) and cap == 'auto':
                cap = auto_patch_size(self.imgs_shape[1:3], self.model_params['n_filter'],
                                      torch.finfo(self.precision).bits // 8, self.device)
            patches = self.__split(imgs, cap)
            self.batch_size = self.__resolve_batch_size(batch_size, patches.shape[0])
            if batch_cap is not None:
                self.batch_size = min(self.batch_size, batch_cap)
            try:
                return self.__predict(patches, self.batch_size, progress_notifier)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if 'out of memory' not in str(exc).lower() or attempt == 3:
                    raise
                shrink_patch = self.batch_size <= 1
                # min(), so the floor cannot enlarge an already smaller patch.
                smaller = tuple(min(c, max(256, c // 2)) for c in self.patch_size)
                if shrink_patch and smaller == tuple(self.patch_size):
                    raise  # at the floor: nothing left to give up
                del patches
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                batch_cap = 1
                if shrink_patch:
                    cap = smaller
                    print(f'Out of memory; retrying with max_patch_size={cap}.')
                else:
                    print('Out of memory; retrying with batch_size=1.')

    def __split(self, imgs, max_patch_size):
        # Plan a uniform tile grid: shrinking the patch below max_patch_size where
        # possible keeps the overlap (and so the redundant compute) small.
        self.patch_size, starts = plan_grid(
            self.imgs_shape[1:3], max_patch_size, 2 * BLEND_MARGIN,
            multiple=16, extra_tiles=self.add_tile)
        self.X_start, self.Y_start = starts
        self.N_x, self.N_y = len(self.X_start), len(self.Y_start)
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_per_img * self.imgs_shape[0]

        imgs = pad_to_patch(imgs, self.patch_size, axes=(1, 2))

        patch_h, patch_w = self.patch_size
        patches = np.empty((self.N, patch_h, patch_w), dtype=imgs.dtype)
        idx = 0
        for t in range(self.imgs_shape[0]):
            for x in self.X_start:
                for y in self.Y_start:
                    patches[idx] = imgs[t, x:x + patch_h, y:y + patch_w]
                    idx += 1
        return patches

    def __resolve_batch_size(self, batch_size, n_patches):
        """Pick a batch size. Only CUDA benefits; MPS and CPU measure flat or worse."""
        if batch_size != 'auto':
            return max(1, int(batch_size))
        if self.device.type != 'cuda':
            return 1
        return auto_batch_size(self.patch_size, self.model_params['n_filter'],
                               torch.finfo(self.precision).bits // 8, self.device,
                               n_patches=n_patches)

    def __predict(self, patches, batch_size=1, progress_notifier=None):
        pin = self.device.type == 'cuda'
        with torch.inference_mode():
            # Initialize result_patches with correct dimensions
            result_patches = {
                target_key: np.zeros(
                    (patches.shape[0], self.model_params['output_heads'][target_key]['channels'], *self.patch_size),
                    dtype='float16'
                )
                for target_key in self.target_keys
            }

            num_batches = int(np.ceil(patches.shape[0] / batch_size))
            _progress_notifier = range(num_batches)
            if self.show_progress and progress_notifier:
                _progress_notifier = progress_notifier.iterator(_progress_notifier)

            for batch_idx in _progress_notifier:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, patches.shape[0])

                # Prepare batch patches
                host = torch.from_numpy(patches[start_idx:end_idx]).to(self.precision)
                if pin:
                    host = host.pin_memory()
                batch_patches = host.to(self.device, non_blocking=pin)
                batch_patches = batch_patches.view(
                    -1, self.model_params['in_channels'], self.patch_size[0], self.patch_size[1]
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
                del batch_patches, batch_results, host

        return result_patches

    def __stitch(self, result_patches):
        """Blend patches back into full frames.

        The blending weights depend only on the tile grid, so they are built once
        by :class:`TileBlender` and shared across every frame, channel and output
        head, and are pre-divided by the coverage so no normalisation pass over the
        full-size result is needed.
        """
        blender = TileBlender(self.imgs_shape[1:3], self.patch_size,
                              (self.X_start, self.Y_start), max_margin=BLEND_MARGIN)
        weights = {(j, k): blender.weight((j, k))
                   for j in range(self.N_x) for k in range(self.N_y)}
        regions = {(j, k): blender.region((j, k))
                   for j in range(self.N_x) for k in range(self.N_y)}

        result = {}
        for target_key in self.target_keys:
            patches_target = result_patches[target_key]
            n_channels = self.model_params['output_heads'][target_key]['channels']
            result_target = np.zeros((self.imgs_shape[0], n_channels) + blender.canvas_shape, dtype='float32')

            for i in range(self.imgs_shape[0]):
                frame_patches = patches_target[i * self.N_per_img:(i + 1) * self.N_per_img]
                for tile, (j, k) in enumerate((j, k) for j in range(self.N_x) for k in range(self.N_y)):
                    x_slice, y_slice = regions[(j, k)]
                    result_target[i, :, x_slice, y_slice] += frame_patches[tile] * weights[(j, k)]

            # Crop to original image size
            result_target = result_target[:, :, :self.imgs_shape[1], :self.imgs_shape[2]]
            result[target_key] = np.squeeze(result_target)

        return result
