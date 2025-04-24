import os
from typing import Union

import numpy as np
import tifffile
import torch

from ..multi_output_unet3d.multi_output_unet3d import MultiOutputUnet3D
from ..progress import ProgressNotifier
from ..utils import get_device


class Predict:
    """Class for prediction of volumetric (3D) data using a 3D U-Net model."""

    def __init__(self, imgs, model_params, result_path=None,
                 network=MultiOutputUnet3D,
                 max_patch_size=(64, 256, 256),  # (depth, height, width)
                 overlap_factor=0.1,
                 batch_size=1,
                 normalization_mode='single',
                 clip_threshold=(0., 99.98),
                 add_tile=0,
                 compress_tif=False,
                 show_progress=True,
                 device: Union[torch.device, str] = 'auto',
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):

        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        if isinstance(imgs, str):
            imgs = tifffile.imread(imgs)

        self.max_patch_size = max_patch_size
        self.overlap_factor = overlap_factor
        self.batch_size = batch_size
        self.add_tile = add_tile
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.result_path = result_path
        self.compress_tif = compress_tif
        self.show_progress = show_progress

        # Preprocess and split data into patches
        imgs = imgs.astype('float32')
        imgs = self.__reshape_data(imgs)
        imgs = self.__preprocess(imgs)
        patches = self.__split(imgs)
        del imgs

        # Load model parameters and initialize model
        self.model_params = torch.load(model_params, map_location=self.device)
        self.model = network(in_channels=self.model_params['in_channels'],
                             n_filter=self.model_params['n_filter'],
                             output_heads=self.model_params['output_heads'],
                             use_interpolation=self.model_params.get('use_interpolation', True)).to(self.device)

        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()

        self.target_keys = list(self.model_params['output_heads'].keys())

        # Predict patches
        result_patches = self.__predict(patches, progress_notifier)
        del patches, self.model

        # Stitch patches back into full volume
        result = self.__stitch(result_patches)
        del result_patches, self.model_params

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

        if torch.cuda.is_available():
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
        if self.normalization_mode == 'single':
            for i in range(len(imgs)):
                img = imgs[i]
                img_clipped = np.clip(img,
                                      np.percentile(img, self.clip_threshold[0]),
                                      np.percentile(img, self.clip_threshold[1]))
                img_normed = (img_clipped - img_clipped.min()) / (img_clipped.ptp() + 1e-8)
                imgs[i] = img_normed

        elif self.normalization_mode in ['first', 'all']:
            reference_img = imgs[0] if self.normalization_mode == 'first' else imgs
            clip_min, clip_max = np.percentile(reference_img, [self.clip_threshold[0],
                                                               self.clip_threshold[1]])
            imgs_clipped = np.clip(imgs, clip_min, clip_max)
            imgs_normed = (imgs_clipped - clip_min) / (clip_max - clip_min + 1e-8)
            imgs[:] = imgs_normed

        else:
            raise ValueError(f'Invalid normalization mode: {self.normalization_mode}')

        return imgs

    def __split(self, imgs):
        N_volumes, D_img, H_img, W_img = imgs.shape
        D_patch, H_patch, W_patch = [
            min(s_img, s_patch) for s_img, s_patch in zip((D_img, H_img, W_img), self.max_patch_size)
        ]

        # Define strides
        stride_dhw = [max(1, int(s * (1 - self.overlap_factor))) for s in (D_patch, H_patch, W_patch)]

        # Calculate starting indices for each dimension
        self.Z_start = list(range(0, max(D_img - D_patch + 1, 1), stride_dhw[0]))
        if self.Z_start[-1] + D_patch < D_img:
            self.Z_start.append(D_img - D_patch)

        self.Y_start = list(range(0, max(H_img - H_patch + 1, 1), stride_dhw[1]))
        if self.Y_start[-1] + H_patch < H_img:
            self.Y_start.append(H_img - H_patch)

        self.X_start = list(range(0, max(W_img - W_patch + 1, 1), stride_dhw[2]))
        if self.X_start[-1] + W_patch < W_img:
            self.X_start.append(W_img - W_patch)

        # Store patch size and counts as attributes for stitching
        self.patch_size = (D_patch, H_patch, W_patch)
        self.N_z = len(self.Z_start)
        self.N_y = len(self.Y_start)
        self.N_x = len(self.X_start)
        self.N_per_vol = self.N_z * self.N_y * self.N_x
        self.imgs_shape = imgs.shape

        patches_list = []

        # Extract patches
        for vol in range(N_volumes):
            for z in self.Z_start:
                for y in self.Y_start:
                    for x in self.X_start:
                        patch = imgs[
                                vol,
                                z: z + D_patch,
                                y: y + H_patch,
                                x: x + W_patch
                                ]
                        patches_list.append(patch)

        patches = np.stack(patches_list)[:, None, ...]  # (N_patches_total, C=1, D,H,W)

        return patches

    def __predict(self, patches, progress_notifier=None):

        results = {k: [] for k in self.target_keys}

        with torch.no_grad():
            num_batches = int(np.ceil(len(patches) / self.batch_size))

            iterator = range(num_batches)

            if progress_notifier and self.show_progress:
                iterator = progress_notifier.iterator(iterator)

            for idx in iterator:
                batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch = torch.tensor(patches[batch_slice], dtype=torch.float32).to(self.device)

                preds = self.model(batch)

                for key in results.keys():
                    results[key].append(preds[key].cpu().numpy())

                del batch, preds

        results = {key: np.concatenate(results[key]) for key in results}

        return results

    def __stitch(self, result_patches, blend_margin=16):
        """
        Stitch 3D patches back into a complete volume with proper overlap handling.

        Parameters:
        -----------
        result_patches : dict
            Dictionary of patches for each output head
        blend_margin : int
            Size of the blending margin for smooth transitions

        Returns:
        --------
        dict
            Dictionary containing stitched volumes for each output head
        """
        result = {}

        # Get dimensions from the original volume shape
        n_volumes, depth, height, width = self.imgs_shape

        for target_key in self.target_keys:
            # Get output channels for this target
            n_channels = self.model_params['output_heads'][target_key]['channels']

            # Initialize result volume and weight map
            result_volume = np.zeros(
                (n_volumes, n_channels, depth, height, width),
                dtype='float32'
            )
            weight_map = np.zeros_like(result_volume)

            # Process each volume separately
            for vol_idx in range(n_volumes):
                # Get patches for this volume
                vol_patches = result_patches[target_key][
                              vol_idx * self.N_per_vol:(vol_idx + 1) * self.N_per_vol
                              ].reshape(self.N_z, self.N_y, self.N_x, n_channels, *self.patch_size)

                # Iterate through all patches
                for z_idx, z_start in enumerate(self.Z_start):
                    for y_idx, y_start in enumerate(self.Y_start):
                        for x_idx, x_start in enumerate(self.X_start):
                            # Get current patch
                            patch = vol_patches[z_idx, y_idx, x_idx]

                            # Create weight mask with smooth transitions at borders
                            patch_weight = np.ones_like(patch)

                            # Apply blending weights at patch edges where overlaps exist
                            if z_idx > 0:  # Front overlap
                                for i in range(min(blend_margin, self.N_z)):
                                    patch_weight[:, i, :, :] = i / blend_margin
                            if z_idx < self.N_z - 1:  # Back overlap
                                for i in range(min(blend_margin, self.N_z)):
                                    patch_weight[:, max(-(i + 1), 0), :, :] = i / blend_margin

                            if y_idx > 0:  # Top overlap
                                for i in range(blend_margin):
                                    patch_weight[:, :, i, :] = i / blend_margin
                            if y_idx < self.N_y - 1:  # Bottom overlap
                                for i in range(blend_margin):
                                    patch_weight[:, :, max(-(i + 1), 0), :] = i / blend_margin

                            if x_idx > 0:  # Left overlap
                                for i in range(blend_margin):
                                    patch_weight[:, :, :, i] = i / blend_margin
                            if x_idx < self.N_x - 1:  # Right overlap
                                for i in range(blend_margin):
                                    patch_weight[:, :, :, max(-(i + 1), 0)] = i / blend_margin

                            # Define slices for placing the patch in the result volume
                            z_slice = slice(z_start, min(z_start + self.patch_size[0], depth))
                            y_slice = slice(y_start, min(y_start + self.patch_size[1], height))
                            x_slice = slice(x_start, min(x_start + self.patch_size[2], width))

                            # Get corresponding patch slice (in case patch extends beyond volume)
                            patch_z_slice = slice(0, z_slice.stop - z_slice.start)
                            patch_y_slice = slice(0, y_slice.stop - y_slice.start)
                            patch_x_slice = slice(0, x_slice.stop - x_slice.start)

                            # Add weighted patch contribution to result
                            result_volume[vol_idx, :, z_slice, y_slice, x_slice] += (
                                    patch[:, patch_z_slice, patch_y_slice, patch_x_slice] *
                                    patch_weight[:, patch_z_slice, patch_y_slice, patch_x_slice]
                            )

                            # Add weights to weight map
                            weight_map[vol_idx, :, z_slice, y_slice, x_slice] += (
                                patch_weight[:, patch_z_slice, patch_y_slice, patch_x_slice]
                            )

            # Normalize by weights to get final result
            # Avoid division by zero
            mask = weight_map > 0
            result_volume[mask] = result_volume[mask] / weight_map[mask]

            # Handle any unweighted areas (should be rare)
            if np.any(~mask):
                result_volume[~mask] = 0

            # Remove singleton dimensions if necessary
            result[target_key] = np.squeeze(result_volume)

        return result
