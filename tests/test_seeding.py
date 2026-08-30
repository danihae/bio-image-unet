"""Reproducibility of DataProcess/Trainer under the opt-in `seed` parameter.

Before this, neither class seeded any RNG: model weight init, DataLoader shuffle
order and the Albumentations augmentation pipeline were all left to whatever
global random state happened to exist. These tests build a tiny synthetic
dataset on disk and confirm that passing the same seed twice reproduces the
augmented tiles, the model's initial weights and the training batch order
exactly, and that leaving seed unset does not change the pre-existing
(unseeded) DataLoader behaviour.
"""
import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from bio_image_unet.unet import DataProcess, Trainer, Unet


def _make_source(tmp_path, n=2, size=40, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    image_dir = tmp_path / 'src_image'
    mask_dir = tmp_path / 'src_mask'
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    for i in range(n):
        img = (rng.rand(size, size) * 255).astype('uint8')
        mask = (rng.rand(size, size) > 0.5).astype('uint8') * 255
        tifffile.imwrite(image_dir / f'{i}.tif', img)
        tifffile.imwrite(mask_dir / f'{i}.tif', mask)
    return str(image_dir) + '/', str(mask_dir) + '/'


def _build_dataset(tmp_path, name, seed):
    image_dir, mask_dir = _make_source(tmp_path)
    data_path = str(tmp_path / name)
    return DataProcess((image_dir, mask_dir), dim_out=(32, 32), aug_factor=3,
                       data_path=data_path, shiftscalerotate=(0.1, 0.1, 15), seed=seed)


def _augmented_tiles(dataset):
    return [(dataset[i]['image'].numpy(), dataset[i]['mask'].numpy()) for i in range(len(dataset))]


def test_dataprocess_seed_reproduces_augmented_tiles(tmp_path):
    tiles_a = _augmented_tiles(_build_dataset(tmp_path, 'data_a', seed=42))
    tiles_b = _augmented_tiles(_build_dataset(tmp_path, 'data_b', seed=42))

    assert len(tiles_a) == len(tiles_b) > 0
    for (img_a, mask_a), (img_b, mask_b) in zip(tiles_a, tiles_b):
        assert np.array_equal(img_a, img_b)
        assert np.array_equal(mask_a, mask_b)


def test_dataprocess_without_seed_is_not_forced_identical(tmp_path):
    tiles_a = _augmented_tiles(_build_dataset(tmp_path, 'data_c', seed=None))
    tiles_b = _augmented_tiles(_build_dataset(tmp_path, 'data_d', seed=None))

    identical = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                    for a, b in zip(tiles_a, tiles_b))
    assert not identical


def test_trainer_seed_reproduces_weights_and_batch_order(tmp_path):
    dataset = _build_dataset(tmp_path, 'data_e', seed=1)

    def snapshot():
        trainer = Trainer(dataset, num_epochs=1, network=Unet, batch_size=2, n_filter=4,
                          save_dir=str(tmp_path / 'save_seeded'), seed=123)
        weights = [p.detach().clone() for p in trainer.model.parameters()]
        first_batch = next(iter(trainer.train_loader))
        return weights, first_batch

    weights_a, batch_a = snapshot()
    weights_b, batch_b = snapshot()

    assert len(weights_a) == len(weights_b) > 0
    for wa, wb in zip(weights_a, weights_b):
        assert torch.equal(wa, wb)
    assert torch.equal(batch_a['image'], batch_b['image'])
    assert torch.equal(batch_a['mask'], batch_b['mask'])


def test_trainer_without_seed_keeps_default_dataloader_behavior(tmp_path):
    dataset = _build_dataset(tmp_path, 'data_f', seed=1)
    trainer = Trainer(dataset, num_epochs=1, network=Unet, batch_size=2, n_filter=4,
                      save_dir=str(tmp_path / 'save_noseed'))

    assert isinstance(trainer.train_loader.sampler, SequentialSampler)
    assert trainer.train_loader.generator is None


def test_trainer_seed_enables_shuffle_with_seeded_generator(tmp_path):
    dataset = _build_dataset(tmp_path, 'data_g', seed=1)
    trainer = Trainer(dataset, num_epochs=1, network=Unet, batch_size=2, n_filter=4,
                      save_dir=str(tmp_path / 'save_shuffle'), seed=99)

    assert isinstance(trainer.train_loader.sampler, RandomSampler)
    assert trainer.train_loader.generator is not None


def test_trainer_deterministic_flag_sets_cudnn_flags(tmp_path):
    dataset = _build_dataset(tmp_path, 'data_h', seed=1)
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    try:
        trainer = Trainer(dataset, num_epochs=1, network=Unet, batch_size=2, n_filter=4,
                          save_dir=str(tmp_path / 'save_det'), seed=7, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert trainer.deterministic is True
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark
