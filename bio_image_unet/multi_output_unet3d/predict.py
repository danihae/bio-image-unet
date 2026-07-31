import os
from typing import Union

import numpy as np
import tifffile
import torch

from ..multi_output_unet3d.multi_output_unet3d import MultiOutputUnet3D
from ..progress import ProgressNotifier
from ..utils import get_device
from ..utils.model_cache import load_cached_model
from ..utils.tiling import TileBlender, auto_patch_size, pad_to_patch, plan_grid

# Ramp width at a seam, per axis. The depth patch is small (a few tens of frames),
# so demanding an in-plane-sized overlap there would multiply the tile count for
# no quality gain; the ramp is capped by the real overlap in any case.
BLEND_MARGIN = (4, 16, 16)


class Predict:
    """Class for prediction of volumetric (3D) data using a 3D U-Net model."""

    def __init__(self, imgs, model_params, result_path=None,
                 network=MultiOutputUnet3D,
                 max_patch_size='auto',  # (depth, height, width), or 'auto'
                 overlap_factor=None,
                 batch_size='auto',
                 normalization_mode='single',
                 clip_threshold=(0., 99.98),
                 add_tile=0,
                 compress_tif=False,
                 show_progress=True,
                 device: Union[torch.device, str] = 'auto',
                 clip_values=None,
                 cache_model=True,
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Predict a volume (or a stack of volumes) patch-wise with a 3D U-Net.

        Parameters
        ----------
        imgs : ndarray or str
            Volume ``(D, H, W)`` or stack of volumes ``(N, D, H, W)``, or a path to a TIFF.
        model_params : str
            Path to the model weights (.pt file).
        result_path : str, optional
            Where to write the result as TIFF. None keeps it in the ``result`` attribute.
        network : Network, optional
            Network architecture. Default is MultiOutputUnet3D.
        max_patch_size : tuple of int or 'auto'
            Maximal patch ``(depth, height, width)``. 'auto' derives it from free
            device memory and the model's filter count, preferring a single patch
            over the whole volume when it fits -- fewer tiles means less redundant
            compute. An explicit tuple is an upper bound: the patch is shrunk below
            it where that lets a whole number of tiles cover the volume with less
            overlap.
        overlap_factor : float, optional
            Deprecated. Overlap now follows the blend margin needed for seamless
            stitching; pass a value only to force a larger minimum overlap.
        batch_size : int or 'auto'
            Patches per forward pass. 'auto' sizes from free GPU memory on CUDA and
            uses 1 elsewhere.
        normalization_mode : str
            'single' (per volume), 'all' (whole stack) or 'first' (first volume).
        clip_threshold : tuple of float
            Percentiles for intensity clipping before prediction.
        add_tile : int, optional
            Additional tiles per axis, increasing the overlap.
        compress_tif : bool, optional
            Compress the written TIFF. Default is False.
        show_progress : bool
            Show a progress bar.
        device : torch.device or str, optional
            Device for the model. 'auto' selects CUDA or MPS if available.
        clip_values : tuple of float, optional
            Absolute intensity limits, bypassing the percentile pass so a caller can
            predict block by block with whole-stack normalization.
        cache_model : bool, optional
            Reuse process-level cached weights across calls. Default is True.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for GUI integration.
        """
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.precision = torch.float16 if self.device.type in ('cuda', 'mps') else torch.float32

        if isinstance(imgs, str):
            imgs = tifffile.imread(imgs)

        self.max_patch_size = max_patch_size
        self.overlap_factor = overlap_factor
        self.add_tile = add_tile
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.clip_values = clip_values
        self.result_path = result_path
        self.compress_tif = compress_tif
        self.show_progress = show_progress

        # Preprocess data (tiling happens after the model is loaded, since an
        # 'auto' patch size depends on the model's filter count)
        imgs = imgs.astype('float32')
        imgs = self.__reshape_data(imgs)
        imgs = self.__preprocess(imgs)

        # Load model parameters and initialize model
        self.model, self.model_params = load_cached_model(
            model_params, network, self.device, self.precision, cache_model,
            from_params={'use_interpolation': True})
        self.target_keys = list(self.model_params['output_heads'].keys())

        # Predict patches
        result_patches = self.__split_and_predict(imgs, batch_size, progress_notifier)
        del imgs
        self.model = None

        # Stitch patches back into full volume
        result = self.__stitch(result_patches)
        del result_patches

        # Save or store results
        if self.result_path is not None:
            for target_key in self.target_keys:
                target_file = self.result_path + target_key + '.tif' if os.path.exists(self.result_path) else (
                        self.result_path + '_' + target_key + '.tif')
                tifffile.imwrite(target_file,
                                 result[target_key],
                                 compression='deflate' if self.compress_tif else None)
            del result
            self.result = None
        else:
            self.result = result

        del self.model_params
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __reshape_data(self, imgs):
        if imgs.ndim == 3:  # Single volume without channel dimension (D,H,W)
            imgs = np.expand_dims(imgs, axis=0)  # Add batch dimension: (1,D,H,W)

        elif imgs.ndim == 4:  # Multiple volumes (N,D,H,W), already correct shape
            pass

        else:
            raise ValueError(f"Unsupported input shape: {imgs.shape}")

        self.imgs_shape = imgs.shape  # (N,D,H,W)

        return imgs

    def __preprocess(self, imgs):
        if self.clip_values is not None:
            lo, hi = self.clip_values
            np.clip(imgs, lo, hi, out=imgs)
            imgs -= lo
            imgs /= (hi - lo + 1e-8)
            return imgs

        if self.normalization_mode == 'single':
            for i in range(len(imgs)):
                img = imgs[i]
                lo, hi = np.percentile(img, [self.clip_threshold[0], self.clip_threshold[1]])
                np.clip(img, lo, hi, out=img)
                img -= lo
                img /= (hi - lo + 1e-8)

        elif self.normalization_mode in ['first', 'all']:
            reference_img = imgs[0] if self.normalization_mode == 'first' else imgs
            clip_min, clip_max = np.percentile(reference_img, [self.clip_threshold[0],
                                                               self.clip_threshold[1]])
            np.clip(imgs, clip_min, clip_max, out=imgs)
            imgs -= clip_min
            imgs /= (clip_max - clip_min + 1e-8)

        else:
            raise ValueError(f'Invalid normalization mode: {self.normalization_mode}')

        return imgs

    def __split_and_predict(self, imgs, batch_size, progress_notifier):
        """Tile, run the model, and shrink the tiles if the device runs out of memory."""
        cap = self.max_patch_size
        for attempt in range(4):
            if isinstance(cap, str) and cap == 'auto':
                cap = auto_patch_size(self.imgs_shape[1:], self.model_params['n_filter'],
                                      torch.finfo(self.precision).bits // 8, self.device)
            patches = self.__split(imgs, cap)
            self.batch_size = self.__resolve_batch_size(batch_size, patches.shape[0])
            try:
                return self.__predict(patches, progress_notifier)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if 'out of memory' not in str(exc).lower() or attempt == 3:
                    raise
                del patches
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                cap = tuple(max(16 if i == 0 else 128, c // 2)
                            for i, c in enumerate(self.patch_size))
                print(f'Out of memory; retrying with max_patch_size={cap}.')

    def __split(self, imgs, max_patch_size):
        min_overlap = [2 * m for m in BLEND_MARGIN]
        if self.overlap_factor:
            # Legacy knob: raise the minimum overlap to the requested fraction.
            min_overlap = [max(mo, int(round(self.overlap_factor * p)))
                           for mo, p in zip(min_overlap, max_patch_size)]

        self.patch_size, starts = plan_grid(
            self.imgs_shape[1:], max_patch_size, min_overlap,
            multiple=16, extra_tiles=self.add_tile)
        self.Z_start, self.Y_start, self.X_start = starts
        self.N_z, self.N_y, self.N_x = (len(s) for s in starts)
        self.N_per_vol = self.N_z * self.N_y * self.N_x

        # Patches may exceed the volume on short axes; pad rather than emit a
        # patch the network cannot pool down (the old code cropped instead, which
        # produced sizes not divisible by 8 and crashed in the decoder).
        imgs = pad_to_patch(imgs, self.patch_size, axes=(1, 2, 3))

        d, h, w = self.patch_size
        n_volumes = self.imgs_shape[0]
        patches = np.empty((n_volumes * self.N_per_vol, 1, d, h, w), dtype=imgs.dtype)
        idx = 0
        for vol in range(n_volumes):
            for z in self.Z_start:
                for y in self.Y_start:
                    for x in self.X_start:
                        patches[idx, 0] = imgs[vol, z:z + d, y:y + h, x:x + w]
                        idx += 1
        return patches

    def __resolve_batch_size(self, batch_size, n_patches):
        """Pick a batch size. Only CUDA benefits; MPS and CPU measure flat or worse."""
        if batch_size != 'auto':
            return max(1, int(batch_size))
        if self.device.type != 'cuda':
            return 1
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return 1
        bytes_per_patch = int(np.prod(self.patch_size)) * (2 if self.precision == torch.float16 else 4)
        return int(np.clip(free * 0.5 // max(bytes_per_patch * 48, 1), 1, min(n_patches, 8)))

    def __predict(self, patches, progress_notifier=None):
        results = {k: [] for k in self.target_keys}
        pin = self.device.type == 'cuda'

        with torch.inference_mode():
            num_batches = int(np.ceil(len(patches) / self.batch_size))

            iterator = range(num_batches)

            if progress_notifier and self.show_progress:
                iterator = progress_notifier.iterator(iterator)

            for idx in iterator:
                batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
                host = torch.from_numpy(patches[batch_slice]).to(self.precision)
                if pin:
                    host = host.pin_memory()
                batch = host.to(self.device, non_blocking=pin)

                preds = self.model(batch)

                for key in results.keys():
                    results[key].append(preds[key].to(torch.float32).cpu().numpy())

                del batch, preds, host

        results = {key: np.concatenate(results[key]) for key in results}

        return results

    def __stitch(self, result_patches):
        """Blend patches back into full volumes.

        The blending weights depend only on the tile grid, so they are built once by
        :class:`TileBlender`, shared across volumes, channels and output heads, and
        pre-divided by the coverage -- no normalisation pass over the full-size
        volume and no unweighted regions.
        """
        n_volumes = self.imgs_shape[0]
        blender = TileBlender(self.imgs_shape[1:], self.patch_size,
                              (self.Z_start, self.Y_start, self.X_start),
                              max_margin=max(BLEND_MARGIN))
        tiles = [(z, y, x) for z in range(self.N_z) for y in range(self.N_y) for x in range(self.N_x)]

        result = {}
        for target_key in self.target_keys:
            n_channels = self.model_params['output_heads'][target_key]['channels']
            volume = np.zeros((n_volumes, n_channels) + blender.canvas_shape, dtype='float32')

            for vol_idx in range(n_volumes):
                vol_patches = result_patches[target_key][
                              vol_idx * self.N_per_vol:(vol_idx + 1) * self.N_per_vol]
                for tile, index in enumerate(tiles):
                    z_slice, y_slice, x_slice = blender.region(index)
                    volume[vol_idx, :, z_slice, y_slice, x_slice] += (
                        vol_patches[tile] * blender.weight(index))

            volume = volume[:, :, :self.imgs_shape[1], :self.imgs_shape[2], :self.imgs_shape[3]]
            result[target_key] = np.squeeze(volume)

        return result
