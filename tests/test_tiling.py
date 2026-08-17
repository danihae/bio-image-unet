"""Tests for the patch-grid planner and the stitching weights."""
import numpy as np
import pytest

from bio_image_unet.utils import tiling
from bio_image_unet.utils.tiling import (ACTIVATION_BYTES_PER_FILTER, TileBlender,
                                         auto_batch_size, auto_patch_size, available_memory,
                                         free_memory, pad_to_patch, plan_axis, plan_grid,
                                         redundancy)

SIZES = list(range(64, 4200, 37))
MAX_PATCHES = (128, 256, 512, 1024)
OVERLAPS = (4, 32, 48)


@pytest.mark.parametrize('max_patch', MAX_PATCHES)
@pytest.mark.parametrize('overlap', OVERLAPS)
def test_plan_axis_covers_without_gaps(max_patch, overlap):
    for size in SIZES:
        patch, starts = plan_axis(size, max_patch, overlap)
        assert patch % 16 == 0, (size, max_patch, patch)
        assert starts[0] == 0
        assert starts == sorted(set(starts)), 'duplicate or unordered starts'
        assert starts[-1] + patch >= size, 'grid stops short of the end'
        assert starts[-1] <= max(size - patch, 0), 'grid runs past the end'
        for a, b in zip(starts, starts[1:]):
            assert b - a <= patch, 'gap between neighbouring patches'


@pytest.mark.parametrize('max_patch', MAX_PATCHES)
def test_plan_axis_no_near_duplicate_tail(max_patch):
    """The tail patch must not sit almost on top of its neighbour.

    Striding by the patch and appending one final patch flush with the end -- the
    obvious way to tile -- yields pairs like starts [0, 230, 256] for a 512 axis,
    so the last two patches share 90% of their pixels.
    """
    for size in SIZES:
        patch, starts = plan_axis(size, max_patch, 32)
        if len(starts) < 2:
            continue
        gaps = np.diff(starts)
        assert gaps.min() >= 0.35 * patch, (size, max_patch, patch, starts)


@pytest.mark.parametrize('max_patch', MAX_PATCHES)
@pytest.mark.parametrize('overlap', OVERLAPS)
def test_plan_axis_redundancy_scales_with_the_overlap(max_patch, overlap):
    """Redundant work should be explained by the requested overlap, nothing more.

    Each internal seam duplicates about ``overlap`` pixels, and rounding the patch
    to a multiple of 16 can add one more step, so the achievable redundancy is
    roughly ``1 + 2*overlap/patch + 16/patch``. Anything beyond that means the grid
    is placing patches badly rather than paying for blending.
    """
    for size in SIZES:
        if size < max_patch:
            continue
        patch, starts = plan_axis(size, max_patch, overlap)
        bound = 1.0 + 2.0 * overlap / patch + 16.0 / patch
        assert redundancy((size,), (patch,), (starts,)) <= bound, (size, max_patch, patch, starts)


@pytest.mark.parametrize('max_patch', (512, 1024))
def test_plan_axis_redundancy_is_small_at_realistic_sizes(max_patch):
    """With the patch sizes actually used, overlap costs a few per cent at most."""
    for size in SIZES:
        if size < max_patch:
            continue
        patch, starts = plan_axis(size, max_patch, 48)
        assert redundancy((size,), (patch,), (starts,)) <= 1.20, (size, max_patch, patch, starts)


def test_plan_axis_beats_naive_on_awkward_sizes():
    """Sizes just above one patch are where the naive scheme is worst."""
    for size in (1200, 1536, 1080, 2304):
        patch, starts = plan_axis(size, 1024, 48)
        naive = int(np.ceil(size / 1024)) * 1024 / size
        assert redundancy((size,), (patch,), (starts,)) < naive


def test_plan_axis_single_patch_for_small_images():
    patch, starts = plan_axis(512, 1024, 48)
    assert (patch, starts) == (512, [0])
    patch, starts = plan_axis(428, 1024, 48)
    assert (patch, starts) == (432, [0]), 'should round up to a multiple of 16 and pad'


def test_plan_grid_matches_plan_axis():
    patch_size, starts = plan_grid((1200, 900), (1024, 1024), 48)
    for axis, (p, s) in enumerate(zip(patch_size, starts)):
        assert (p, s) == plan_axis((1200, 900)[axis], 1024, 48)


def test_plan_grid_accepts_per_axis_overlap():
    patch_size, starts = plan_grid((100, 512, 512), (32, 256, 256), [4, 32, 32])
    assert len(patch_size) == 3 and len(starts) == 3
    assert patch_size[0] % 16 == 0


def test_pad_to_patch_handles_padding_larger_than_axis():
    """np.pad cannot reflect further than the axis length, so tiny axes need edge mode."""
    arr = np.arange(3 * 4, dtype='float32').reshape(1, 3, 4)
    out = pad_to_patch(arr, (16, 16), axes=(1, 2))
    assert out.shape == (1, 16, 16)
    assert np.isfinite(out).all()


def test_pad_to_patch_is_a_noop_when_large_enough():
    arr = np.zeros((2, 64, 64), dtype='float32')
    assert pad_to_patch(arr, (32, 32), axes=(1, 2)) is arr


class TestTileBlender:
    def test_weights_form_a_partition_of_unity(self):
        for shape, max_patch in [((1200, 900), (1024, 1024)), ((512, 512), (256, 256)),
                                 ((2048, 2048), (1024, 1024))]:
            patch_size, starts = plan_grid(shape, max_patch, 48)
            blender = TileBlender(shape, patch_size, starts, max_margin=24)
            total = np.zeros(blender.canvas_shape, dtype='float64')
            for index in np.ndindex(*(len(s) for s in starts)):
                total[blender.region(index)] += blender.weight(index)
            assert np.allclose(total, 1.0, atol=1e-5), shape

    def test_no_unweighted_pixels_at_exact_multiples(self):
        """A 2048 axis tiled by 1024 is where hard-margin stitching leaves a hole."""
        shape = (2048, 2048)
        patch_size, starts = plan_grid(shape, (1024, 1024), 48)
        blender = TileBlender(shape, patch_size, starts, max_margin=24)
        total = np.zeros(blender.canvas_shape, dtype='float64')
        for index in np.ndindex(*(len(s) for s in starts)):
            total[blender.region(index)] += blender.weight(index)
        assert (total > 0).all()

    def test_reconstructs_a_constant_field(self):
        shape = (1200, 1200)
        patch_size, starts = plan_grid(shape, (1024, 1024), 48)
        blender = TileBlender(shape, patch_size, starts, max_margin=24)
        canvas = np.zeros(blender.canvas_shape, dtype='float32')
        for index in np.ndindex(*(len(s) for s in starts)):
            canvas[blender.region(index)] += 7.0 * blender.weight(index)
        assert np.allclose(canvas[:shape[0], :shape[1]], 7.0, atol=1e-4)

    def test_three_dimensional_weights(self):
        shape = (64, 300, 300)
        patch_size, starts = plan_grid(shape, (32, 256, 256), [4, 32, 32])
        blender = TileBlender(shape, patch_size, starts, max_margin=16)
        total = np.zeros(blender.canvas_shape, dtype='float64')
        for index in np.ndindex(*(len(s) for s in starts)):
            total[blender.region(index)] += blender.weight(index)
        assert np.allclose(total, 1.0, atol=1e-5)

    def test_cache_is_bounded(self):
        shape = (64, 512, 512)
        patch_size, starts = plan_grid(shape, (32, 256, 256), [4, 32, 32])
        blender = TileBlender(shape, patch_size, starts, max_cache_bytes=1024)
        for index in np.ndindex(*(len(s) for s in starts)):
            assert blender.weight(index).shape == patch_size
        assert blender._cache_bytes <= 1024


class TestAutoPatchSize:
    def test_never_exceeds_the_image(self):
        import torch
        device = torch.device('cpu')
        for shape in [(512, 512), (1200, 1200), (2000, 2000), (200, 1024)]:
            patch = auto_patch_size(shape, 64, 2, device)
            for p, s in zip(patch, shape):
                assert p <= ((s + 15) // 16) * 16

    def test_shrinks_when_memory_is_scarce(self):
        import torch
        device = torch.device('cpu')
        roomy = auto_patch_size((4096, 4096), 64, 2, device, memory_fraction=0.9)
        tight = auto_patch_size((4096, 4096), 64, 2, device, memory_fraction=0.001)
        assert np.prod(tight) < np.prod(roomy)

    def test_respects_min_side(self):
        import torch
        patch = auto_patch_size((4096, 4096), 64, 4, torch.device('cpu'),
                                memory_fraction=1e-6, min_side=(256, 256))
        assert min(patch) >= 256

    def test_budget_does_not_track_free_memory(self):
        """The budget must be a stable device property.

        It was briefly derived from psutil's *available* memory, which drifts with
        whatever else the machine is doing. That silently changed the tile grid --
        and so the result at seams -- between two runs on the same input.
        """
        import torch
        for device in (torch.device('cpu'), torch.device('mps'), torch.device('cuda')):
            try:
                first = available_memory(device)
            except Exception:
                continue
            assert first == available_memory(device)
            assert first > 0

    def test_plan_is_reproducible(self):
        import torch
        device = torch.device('cpu')
        for shape, n_filter in [((1200, 1200), 64), ((2000, 2000), 64), ((64, 428, 472), 16)]:
            plans = {auto_patch_size(shape, n_filter, 2, device) for _ in range(5)}
            assert len(plans) == 1, (shape, plans)

    def test_derived_side_is_quantised(self):
        """The derived side lands on a coarse grid, not on every integer.

        This is robustness, not a guarantee: a budget that moves far enough still
        moves the plan, and two machines with different memory legitimately plan
        differently. Pin ``max_patch_size`` when a run has to reproduce elsewhere.
        """
        import torch
        device = torch.device('cpu')
        sides = set()
        for fraction in (0.05, 0.10, 0.15, 0.20, 0.30, 0.45):
            patch = auto_patch_size((4096, 4096), 64, 2, device, memory_fraction=fraction)
            for side in patch:
                if side < 4096:  # only the memory-derived sides are quantised
                    assert side % 64 == 0, (fraction, patch)
            sides.add(patch)
        assert len(sides) > 1, 'the budget should still influence the plan'

    def test_plan_is_monotonic_in_the_budget(self):
        """More memory must never produce a smaller patch."""
        import torch
        device = torch.device('cpu')
        previous = None
        for fraction in (0.02, 0.05, 0.10, 0.20, 0.40, 0.80):
            patch = auto_patch_size((4096, 4096), 64, 2, device, memory_fraction=fraction)
            if previous is not None:
                assert all(new >= old for new, old in zip(patch, previous)), (fraction, patch, previous)
            previous = patch


def _sample_bytes(patch, n_filter, itemsize):
    """What one patch of activations costs, by the model both planners share."""
    return int(np.prod(patch)) * ACTIVATION_BYTES_PER_FILTER * n_filter * itemsize


class TestAutoBatchSize:
    CARDS = (6 << 30, 8 << 30, 12 << 30, 24 << 30, 32 << 30, 80 << 30)

    def _pin_memory(self, monkeypatch, total):
        for name in ('available_memory', 'free_memory'):
            monkeypatch.setattr(tiling, name, lambda device, _t=total: _t)

    def test_patch_and_batch_do_not_multiply(self, monkeypatch):
        """The regression this function exists for.

        The patch spends its whole fraction of the device on one sample, so a
        batch budgeted independently multiplied that: a 32 GiB card used to ask
        for over 100 GiB on any multi-frame stack.
        """
        import torch
        device = torch.device('cpu')
        for total in self.CARDS:
            self._pin_memory(monkeypatch, total)
            patch = auto_patch_size((4096, 4096), 64, 2, device)
            batch = auto_batch_size(patch, 64, 2, device, n_patches=500)
            demand = batch * _sample_bytes(patch, 64, 2)
            assert demand <= total, (total, patch, batch, demand / total)

    def test_stays_within_its_share_of_free_memory(self, monkeypatch):
        import torch
        device = torch.device('cpu')
        total = 32 << 30
        self._pin_memory(monkeypatch, total)
        for patch in [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (32, 256, 256)]:
            n_filter = 16 if len(patch) == 3 else 64
            batch = auto_batch_size(patch, n_filter, 2, device, n_patches=1000)
            assert batch >= 1
            if batch > 1:  # a forced batch of 1 may exceed the share; nothing smaller exists
                assert batch * _sample_bytes(patch, n_filter, 2) <= total * 0.6

    def test_shrinks_as_the_patch_grows(self, monkeypatch):
        """A bigger patch must never buy a bigger batch."""
        import torch
        device = torch.device('cpu')
        self._pin_memory(monkeypatch, 32 << 30)
        previous = None
        for side in (128, 256, 512, 1024, 2048):
            batch = auto_batch_size((side, side), 64, 2, device, n_patches=1000)
            if previous is not None:
                assert batch <= previous, (side, batch, previous)
            previous = batch

    def test_tracks_free_memory_not_installed(self, monkeypatch):
        """A card another process is holding must yield a smaller batch."""
        import torch
        device = torch.device('cpu')
        monkeypatch.setattr(tiling, 'available_memory', lambda device: 32 << 30)
        monkeypatch.setattr(tiling, 'free_memory', lambda device: 32 << 30)
        idle = auto_batch_size((512, 512), 64, 2, device, n_patches=1000, max_batch=32)
        monkeypatch.setattr(tiling, 'free_memory', lambda device: 2 << 30)
        busy = auto_batch_size((512, 512), 64, 2, device, n_patches=1000, max_batch=32)
        assert busy < idle

    def test_is_quantised_against_drifting_free_memory(self, monkeypatch):
        """Drift in free memory must not change the batch, and with it the kernels."""
        import torch
        device = torch.device('cpu')
        monkeypatch.setattr(tiling, 'available_memory', lambda device: 32 << 30)
        seen = set()
        for jitter in (1.00, 0.97, 0.94, 0.91):
            monkeypatch.setattr(tiling, 'free_memory', lambda device, _j=jitter: int((32 << 30) * _j))
            batch = auto_batch_size((512, 512), 64, 2, device, n_patches=1000, max_batch=32)
            assert batch & (batch - 1) == 0, f'{batch} is not a power of two'
            seen.add(batch)
        assert len(seen) == 1, f'a few percent of drift changed the batch: {seen}'

        monkeypatch.setattr(tiling, 'free_memory', lambda device: 4 << 30)
        assert auto_batch_size((512, 512), 64, 2, device, n_patches=1000, max_batch=32) < min(seen)

    def test_never_returns_less_than_one(self, monkeypatch):
        import torch
        device = torch.device('cpu')
        self._pin_memory(monkeypatch, 1 << 20)
        assert auto_batch_size((2048, 2048), 64, 4, device, n_patches=1000) == 1
        assert auto_batch_size((2048, 2048), 64, 4, device, n_patches=0) == 1

    def test_never_exceeds_the_work_or_the_ceiling(self, monkeypatch):
        import torch
        device = torch.device('cpu')
        self._pin_memory(monkeypatch, 512 << 30)
        assert auto_batch_size((64, 64), 64, 2, device, n_patches=3) == 3
        assert auto_batch_size((64, 64), 64, 2, device, n_patches=1000) == 4
        assert auto_batch_size((64, 64), 64, 2, device, n_patches=1000, max_batch=8) == 8
        assert auto_batch_size((64, 64), 64, 2, device) == 4

    def test_free_memory_falls_back_to_available_off_cuda(self):
        import torch
        device = torch.device('cpu')
        assert free_memory(device) == available_memory(device)
