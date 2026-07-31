"""Shape and dtype round-trips through the multi-output predictors.

These use tiny randomly-initialised checkpoints rather than trained weights: the
point is that every input shape survives tiling, prediction and stitching, in
particular shapes smaller than the patch, which used to reach the network with
sizes the decoder could not pool down.
"""
import numpy as np
import pytest
import torch

from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.multi_output_unet.predict import Predict as Predict2D
from bio_image_unet.multi_output_unet3d.multi_output_unet3d import MultiOutputUnet3D
from bio_image_unet.multi_output_unet3d.predict import Predict as Predict3D
from bio_image_unet.progress import ProgressNotifier

DEVICE = torch.device('cpu')
HEADS_2D = {'mask': {'channels': 1, 'activation': 'sigmoid'},
            'vec': {'channels': 2, 'activation': None}}
HEADS_3D = {'mask': {'channels': 1, 'activation': 'sigmoid'}}
SILENT = ProgressNotifier.silent_notifier()


def checkpoint_2d(n_filter=4, deep_supervision=True):
    model = MultiOutputNestedUNet_3Levels(in_channels=1, n_filter=n_filter,
                                          output_heads=HEADS_2D,
                                          deep_supervision=deep_supervision, train_mode=False)
    return {'in_channels': 1, 'n_filter': n_filter, 'output_heads': HEADS_2D,
            'deep_supervision': deep_supervision, 'state_dict': model.state_dict()}


def checkpoint_3d(n_filter=4):
    model = MultiOutputUnet3D(in_channels=1, n_filter=n_filter, output_heads=HEADS_3D,
                              use_interpolation=True)
    return {'in_channels': 1, 'n_filter': n_filter, 'output_heads': HEADS_3D,
            'use_interpolation': True, 'state_dict': model.state_dict()}


@pytest.fixture(scope='module')
def ckpt_2d(tmp_path_factory):
    path = tmp_path_factory.mktemp('models') / 'tiny2d.pt'
    torch.save(checkpoint_2d(), path)
    return str(path)


@pytest.fixture(scope='module')
def ckpt_3d(tmp_path_factory):
    path = tmp_path_factory.mktemp('models') / 'tiny3d.pt'
    torch.save(checkpoint_3d(), path)
    return str(path)


def predict_2d(imgs, ckpt, **kwargs):
    kwargs.setdefault('max_patch_size', (64, 64))
    kwargs.setdefault('normalization_mode', 'all')
    return Predict2D(imgs, model_params=ckpt, result_path=None,
                     network=MultiOutputNestedUNet_3Levels,
                     device=DEVICE, progress_notifier=SILENT, **kwargs).result


def predict_3d(imgs, ckpt, **kwargs):
    kwargs.setdefault('max_patch_size', (16, 64, 64))
    kwargs.setdefault('normalization_mode', 'all')
    return Predict3D(imgs, model_params=ckpt, result_path=None,
                     device=DEVICE, progress_notifier=SILENT, **kwargs).result


class TestPredict2D:
    @pytest.mark.parametrize('shape', [(48, 48), (64, 64), (100, 100), (128, 128), (150, 90)])
    def test_single_image_shapes(self, ckpt_2d, shape):
        result = predict_2d(np.random.rand(*shape).astype('float32'), ckpt_2d)
        assert result['mask'].shape == shape
        assert result['vec'].shape == (2,) + shape

    @pytest.mark.parametrize('shape', [(3, 48, 48), (2, 100, 140), (5, 64, 64)])
    def test_stack_shapes(self, ckpt_2d, shape):
        result = predict_2d(np.random.rand(*shape).astype('float32'), ckpt_2d)
        assert result['mask'].shape == shape
        assert result['vec'].shape == (shape[0], 2) + shape[1:]

    def test_smaller_than_patch(self, ckpt_2d):
        """A 20x20 image with a 64x64 patch must be padded, not fed in at 20x20."""
        result = predict_2d(np.random.rand(2, 20, 20).astype('float32'), ckpt_2d)
        assert result['mask'].shape == (2, 20, 20)
        assert np.isfinite(result['mask']).all()

    def test_auto_patch_size(self, ckpt_2d):
        result = predict_2d(np.random.rand(2, 96, 96).astype('float32'), ckpt_2d,
                            max_patch_size='auto')
        assert result['mask'].shape == (2, 96, 96)

    def test_uint16_input_is_accepted(self, ckpt_2d):
        imgs = (np.random.rand(2, 64, 64) * 4000).astype('uint16')
        result = predict_2d(imgs, ckpt_2d)
        assert result['mask'].shape == (2, 64, 64)

    def test_probabilities_are_in_range(self, ckpt_2d):
        result = predict_2d(np.random.rand(2, 100, 100).astype('float32'), ckpt_2d)
        assert result['mask'].min() >= 0.0 and result['mask'].max() <= 1.0

    def test_clip_values_match_whole_stack_normalization(self, ckpt_2d):
        """Block-wise prediction must reproduce a whole-stack 'all' run exactly."""
        imgs = (np.random.rand(6, 64, 64) * 1000).astype('uint16')
        whole = predict_2d(imgs, ckpt_2d, normalization_mode='all')
        lo, hi = float(np.nanpercentile(imgs, 0.)), float(np.percentile(imgs, 99.98))
        blocks = [predict_2d(imgs[i:i + 2], ckpt_2d, clip_values=(lo, hi)) for i in (0, 2, 4)]
        stitched = np.concatenate([b['mask'] for b in blocks])
        assert np.allclose(whole['mask'], stitched, atol=1e-5)

    def test_batch_size_does_not_change_the_result(self, ckpt_2d):
        imgs = np.random.rand(4, 64, 64).astype('float32')
        one = predict_2d(imgs, ckpt_2d, batch_size=1)
        many = predict_2d(imgs, ckpt_2d, batch_size=4)
        assert np.allclose(one['mask'], many['mask'], atol=1e-6)

    def test_tiled_matches_untiled_away_from_seams(self, ckpt_2d):
        imgs = np.random.rand(1, 128, 128).astype('float32')
        untiled = predict_2d(imgs, ckpt_2d, max_patch_size=(128, 128))['mask']
        tiled = predict_2d(imgs, ckpt_2d, max_patch_size=(80, 80))['mask']
        assert untiled.shape == tiled.shape
        # Tile interiors should agree closely; seams are blended so they differ a little.
        assert np.abs(untiled - tiled).mean() < 0.05

    def test_prune_level_changes_cost_not_shape(self, ckpt_2d):
        imgs = np.random.rand(2, 64, 64).astype('float32')
        for level in (1, 2, 3):
            result = predict_2d(imgs, ckpt_2d, prune_level=level)
            assert result['mask'].shape == (2, 64, 64)

    def test_prune_level_requires_deep_supervision(self, tmp_path):
        path = tmp_path / 'nods.pt'
        torch.save(checkpoint_2d(deep_supervision=False), path)
        with pytest.raises(ValueError, match='deep supervision'):
            predict_2d(np.random.rand(1, 64, 64).astype('float32'), str(path), prune_level=2)

    def test_rejects_invalid_prune_level(self, ckpt_2d):
        with pytest.raises(ValueError, match='prune_level'):
            predict_2d(np.random.rand(1, 64, 64).astype('float32'), ckpt_2d, prune_level=7)


class TestPredict3D:
    @pytest.mark.parametrize('shape', [(16, 64, 64), (32, 64, 64), (24, 80, 96)])
    def test_volume_shapes(self, ckpt_3d, shape):
        result = predict_3d(np.random.rand(*shape).astype('float32'), ckpt_3d)
        assert result['mask'].shape == shape

    def test_depth_not_a_multiple_of_eight(self, ckpt_3d):
        """Regression: a 30-frame stack used to reach the decoder at depth 30 and crash."""
        result = predict_3d(np.random.rand(30, 40, 96).astype('float32'), ckpt_3d)
        assert result['mask'].shape == (30, 40, 96)
        assert np.isfinite(result['mask']).all()

    def test_smaller_than_patch_on_every_axis(self, ckpt_3d):
        result = predict_3d(np.random.rand(10, 20, 20).astype('float32'), ckpt_3d)
        assert result['mask'].shape == (10, 20, 20)

    def test_auto_patch_size(self, ckpt_3d):
        result = predict_3d(np.random.rand(20, 64, 64).astype('float32'), ckpt_3d,
                            max_patch_size='auto')
        assert result['mask'].shape == (20, 64, 64)

    def test_probabilities_are_in_range(self, ckpt_3d):
        result = predict_3d(np.random.rand(16, 64, 64).astype('float32'), ckpt_3d)
        assert result['mask'].min() >= 0.0 and result['mask'].max() <= 1.0


def test_model_cache_reuses_weights(ckpt_2d):
    from bio_image_unet.utils.model_cache import clear_model_cache, load_cached_model

    clear_model_cache()
    imgs = np.random.rand(1, 64, 64).astype('float32')
    predict_2d(imgs, ckpt_2d)
    first, _ = load_cached_model(ckpt_2d, MultiOutputNestedUNet_3Levels, DEVICE, torch.float32,
                                True, from_params={'deep_supervision': False}, train_mode=False)
    second, _ = load_cached_model(ckpt_2d, MultiOutputNestedUNet_3Levels, DEVICE, torch.float32,
                                  True, from_params={'deep_supervision': False}, train_mode=False)
    assert first is second
    clear_model_cache()
