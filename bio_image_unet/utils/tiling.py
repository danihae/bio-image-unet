"""Tile planning for patch-wise prediction.

Prediction splits an image or volume into patches, runs the network on each and
stitches the results back together. How the patches are placed decides how much
redundant work the network does: every pixel covered by two patches is predicted
twice.

The planner here places a uniform grid of patches that covers the axis exactly,
overlapping by at least ``min_overlap`` so the stitcher has something to blend,
and shrinks the patch below ``max_patch`` when that reduces the redundancy. The
naive alternative -- take the largest legal patch, stride by it, and append one
extra patch flush with the end -- can process up to three times the pixels of
the image, because the appended patch nearly duplicates its neighbour.
"""
from typing import List, Tuple

import numpy as np

__all__ = ['plan_axis', 'plan_grid', 'pad_to_patch', 'redundancy', 'TileBlender',
           'available_memory', 'free_memory', 'auto_patch_size', 'auto_batch_size',
           'ACTIVATION_BYTES_PER_FILTER']

# Peak activation memory of these U-Nets, in bytes per pixel of patch, per filter
# of the first level, per byte of the compute dtype. Measured at ~21 for both the
# 2D nested U-Net (n_filter=64: 2.7 kB/px) and the 3D U-Net (n_filter=16:
# 0.66 kB/voxel) -- the footprint is dominated by the full-resolution feature maps,
# so it is close to linear in n_filter for this family.
ACTIVATION_BYTES_PER_FILTER = 21


# Granularity the auto-derived patch side is snapped to, so that a slightly
# different memory reading does not change the tile grid.
QUANTUM = 64


def _round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""
    return int(((value + multiple - 1) // multiple) * multiple)


def _round_down(value: int, multiple: int) -> int:
    """Round ``value`` down to a multiple of ``multiple``."""
    return int((value // multiple) * multiple)


def plan_axis(size: int, max_patch: int, min_overlap: int = 32, multiple: int = 16,
              extra_tiles: int = 0) -> Tuple[int, List[int]]:
    """Plan the patch size and start offsets covering one axis.

    Parameters
    ----------
    size : int
        Length of the axis in pixels.
    max_patch : int
        Largest patch permitted along this axis (memory / model limit).
    min_overlap : int, optional
        Minimum overlap in pixels between neighbouring patches. The stitcher
        discards or down-weights a margin at every internal patch border, so the
        overlap must be at least twice that margin for the tiles to still cover
        the axis jointly. Default is 32.
    multiple : int, optional
        The patch size is rounded up to a multiple of this, as required by the
        network's pooling depth. Default is 16.
    extra_tiles : int, optional
        Add this many patches beyond the minimum, increasing the overlap.
        Default is 0.

    Returns
    -------
    patch : int
        Patch length along this axis. May exceed ``size``, in which case the
        caller must pad the image (see :func:`pad_to_patch`).
    starts : list of int
        Start offsets, ascending, first 0 and last ``size - patch``. Consecutive
        patches overlap by at least ``min_overlap`` where the axis is long
        enough to allow it.
    """
    if size < 1:
        raise ValueError(f'axis size must be positive, got {size}')
    cap = _round_up(min(size, max_patch), multiple)

    if cap >= size and extra_tiles == 0:
        # One patch covers the axis; the caller reflects the image up to `cap`.
        return cap, [0]

    overlap = int(min(min_overlap, cap // 2))
    n = int(np.ceil((size - overlap) / float(cap - overlap))) if cap > overlap else 1
    n = max(n, 2) + extra_tiles

    # With the tile count fixed, use the smallest patch that still covers the
    # axis with the required overlap -- this is where the redundancy is saved.
    patch = min(_round_up(int(np.ceil((size + (n - 1) * overlap) / float(n))), multiple), cap)
    if n * patch < size:  # only reachable if `cap` itself is too small; widen the grid
        n = int(np.ceil(size / float(patch)))

    starts = sorted({int(round(s)) for s in np.linspace(0, max(size - patch, 0), n)})
    return patch, starts


def plan_grid(shape, max_patch_size, min_overlap, multiple: int = 16, extra_tiles: int = 0):
    """Plan :func:`plan_axis` over every axis of ``shape``.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape to cover, e.g. ``(H, W)`` or ``(D, H, W)``.
    max_patch_size : sequence of int
        Per-axis patch limits, same length as ``shape``.
    min_overlap : int or sequence of int
        Minimum overlap per axis; a scalar applies to every axis. A short axis
        (such as time in a 3D stack) usually wants a smaller value than the
        in-plane axes, since its patch is small to begin with.
    multiple : int, optional
        Patch sizes are rounded up to a multiple of this. Default is 16.
    extra_tiles : int, optional
        Extra patches per axis. Default is 0.

    Returns
    -------
    patch_size : tuple of int
        Patch shape.
    starts : list of list of int
        Start offsets per axis.
    """
    if np.isscalar(min_overlap):
        min_overlap = [int(min_overlap)] * len(shape)
    planned = [plan_axis(s, m, o, multiple, extra_tiles)
               for s, m, o in zip(shape, max_patch_size, min_overlap)]
    return tuple(p for p, _ in planned), [st for _, st in planned]


def pad_to_patch(arr: np.ndarray, patch_size, axes) -> np.ndarray:
    """Pad ``arr`` along ``axes`` so every axis is at least its patch length.

    Reflection padding is used where the array is long enough for it, edge
    padding otherwise (``np.pad`` cannot reflect further than the axis length).

    Parameters
    ----------
    arr : np.ndarray
        Array to pad.
    patch_size : sequence of int
        Required minimum length per padded axis.
    axes : sequence of int
        Axes of ``arr`` that ``patch_size`` refers to.

    Returns
    -------
    np.ndarray
        The padded array, or ``arr`` unchanged if no padding is needed.
    """
    pad = [(0, 0)] * arr.ndim
    mode = 'reflect'
    needed = False
    for axis, patch in zip(axes, patch_size):
        extra = max(patch - arr.shape[axis], 0)
        if extra:
            needed = True
            if extra >= arr.shape[axis]:
                mode = 'edge'
            pad[axis] = (0, extra)
    return np.pad(arr, pad, mode=mode) if needed else arr


def available_memory(device) -> int:
    """Memory budget for prediction on ``device``, in bytes.

    This is deliberately a *stable* figure -- installed device memory, not
    momentarily free memory. The budget decides the patch size, the patch size
    decides the tile grid, and the tile grid changes the result at seams, so a
    budget that drifted with transient memory pressure would make predictions
    irreproducible from one run to the next. Headroom comes from the fraction the
    caller applies and from retrying smaller after an out-of-memory error.

    Parameters
    ----------
    device : torch.device
        Device the model runs on.

    Returns
    -------
    int
        Total device memory on CUDA, the recommended maximum on MPS (unified
        memory, so a share of RAM), or a share of system RAM on CPU.
    """
    import torch

    if device.type == 'cuda':
        try:
            return int(torch.cuda.get_device_properties(device).total_memory)
        except Exception:
            return 2 << 30
    if device.type == 'mps':
        try:
            return int(torch.mps.recommended_max_memory())
        except (AttributeError, RuntimeError):
            pass
    try:
        import psutil
        # Unified memory on Apple silicon: the GPU may use most of RAM. On CPU the
        # working set competes with everything else, so claim less.
        return int(psutil.virtual_memory().total * (0.75 if device.type == 'mps' else 0.5))
    except Exception:
        return 4 << 30


def free_memory(device) -> int:
    """Memory momentarily free on ``device``, in bytes.

    Unlike :func:`available_memory` this may drift between runs, which is fine for
    a batch size -- the batch leaves the tile grid, and so the result, alone -- but
    not for a patch size.

    Parameters
    ----------
    device : torch.device
        Device the model runs on.

    Returns
    -------
    int
        Free memory on CUDA, else :func:`available_memory`.
    """
    import torch

    if device.type == 'cuda':
        try:
            return int(torch.cuda.mem_get_info(device)[0])
        except Exception:
            pass
    return available_memory(device)


def auto_batch_size(patch_size, n_filter: int, itemsize: int, device,
                    n_patches: int = None, memory_fraction: float = 0.6,
                    max_batch: int = 4) -> int:
    """How many patches of ``patch_size`` fit on the device at once.

    Costed with the same activation model as :func:`auto_patch_size`, on top of
    the patch rather than instead of it: an auto-sized patch already fills its
    share of the device, so a batch of 1 is the normal answer there.

    Parameters
    ----------
    patch_size : sequence of int
        Patch shape the grid was planned with, e.g. ``(H, W)`` or ``(D, H, W)``.
    n_filter : int
        The model's first-level filter count, from its checkpoint.
    itemsize : int
        Bytes per element of the compute dtype (2 for float16, 4 for float32).
    device : torch.device
        Device the model runs on.
    n_patches : int, optional
        Total patches to predict; the batch never exceeds the work there is.
        Default is None, meaning no such limit.
    memory_fraction : float, optional
        Share of free memory the batch may occupy. Default is 0.6.
    max_batch : int, optional
        Ceiling on the returned batch. Default is 4: measured on CUDA, batching
        pays only for patches below ~256 px (1.3x at 128 px, peaking at 4) and
        costs throughput above that.

    Returns
    -------
    int
        Batch size, at least 1.
    """
    elements = 1
    for side in patch_size:
        elements *= int(side)
    bytes_per_sample = max(elements * ACTIVATION_BYTES_PER_FILTER * int(n_filter) * int(itemsize), 1)
    affordable = max(int(free_memory(device) * memory_fraction) // bytes_per_sample, 1)
    # Quantised, so that drift in free memory cannot change the batch and with it
    # the choice of kernels. The n_patches cap below is exact, not quantised.
    quantised = 1 << (int(affordable).bit_length() - 1)

    ceiling = max(int(max_batch), 1)
    if n_patches is not None:
        ceiling = min(ceiling, max(int(n_patches), 1))
    return int(np.clip(quantised, 1, ceiling))


def auto_patch_size(shape, n_filter: int, itemsize: int, device,
                    memory_fraction: float = 0.35, min_side=None, max_side: int = 2048,
                    multiple: int = 16):
    """Largest patch the device can run for this image, as a per-axis tuple.

    A patch covering the whole image needs no tiling at all, and the fewer tiles a
    plan uses the less redundant work the network does -- so the useful patch size
    is "as large as memory allows, but no larger than the image". Throughput per
    pixel is flat above roughly 512 px in-plane and falls off below it, hence
    ``min_side``.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape of the image or volume, e.g. ``(H, W)`` or ``(D, H, W)``.
    n_filter : int
        The model's first-level filter count, from its checkpoint.
    itemsize : int
        Bytes per element of the compute dtype (2 for float16, 4 for float32).
    device : torch.device
        Device the model runs on.
    memory_fraction : float, optional
        Share of available memory to spend on one patch. Default is 0.35, which
        leaves room for the weights, the host-side buffers and fragmentation.
    min_side : sequence of int, optional
        Per-axis floor for the patch. Defaults to 256 on in-plane axes and 16 on
        the leading axis of a 3D volume.
    max_side : int, optional
        Per-axis ceiling. Default is 2048.
    multiple : int, optional
        Patch sizes are rounded to a multiple of this. Default is 16.

    Returns
    -------
    tuple of int
        Patch size per axis, never larger than the image.

    Notes
    -----
    The result is deterministic for a given device, model and image, because the
    budget is installed memory rather than momentarily free memory. It is not
    identical across machines with different memory, and the tile grid changes the
    result slightly at seams -- pass an explicit ``max_patch_size`` when a run has
    to reproduce exactly elsewhere.
    """
    shape = tuple(int(s) for s in shape)
    if min_side is None:
        min_side = (16,) + (256,) * (len(shape) - 1) if len(shape) == 3 else (256,) * len(shape)

    bytes_per_element = max(ACTIVATION_BYTES_PER_FILTER * int(n_filter) * int(itemsize), 1)
    budget_elements = max(int(available_memory(device) * memory_fraction) // bytes_per_element, 1)

    total = 1
    for s in shape:
        total *= s
    if total <= budget_elements:
        scale = 1.0
    else:
        scale = (budget_elements / float(total)) ** (1.0 / len(shape))

    patch = []
    for s, floor in zip(shape, min_side):
        # Quantise the derived side coarsely so that a small difference in the
        # reported budget -- another torch version, a different machine of the
        # same class -- cannot silently produce a different tile grid.
        side = max(_round_down(int(s * scale), QUANTUM), multiple)
        side = int(np.clip(side, min(floor, s), max_side))
        patch.append(min(_round_up(s, multiple), side))
    return tuple(patch)


def _axis_profiles(patch: int, starts: List[int], max_margin: int) -> List[np.ndarray]:
    """Linear ramps into each internal seam, one profile per patch along one axis."""
    profiles = []
    for j, start in enumerate(starts):
        w = np.ones(patch, dtype=np.float32)
        if j > 0:
            m = min(starts[j - 1] + patch - start, patch, max_margin)
            if m > 0:
                w[:m] = np.arange(1, m + 1, dtype=np.float32) / (m + 1)
        if j < len(starts) - 1:
            m = min(start + patch - starts[j + 1], patch, max_margin)
            if m > 0:
                w[patch - m:] = np.arange(m, 0, -1, dtype=np.float32) / (m + 1)
        profiles.append(w)
    return profiles


class TileBlender:
    """Per-tile blending weights for stitching, normalised so contributions sum to one.

    The weight of a tile is the outer product of one linear ramp per axis, tapering
    into every seam it shares with a neighbour. Because the same weights apply to
    every frame, every channel and every output head, they are computed once here
    and divided by the accumulated coverage up front -- so stitching is a plain
    multiply-accumulate with no final division and no zero-weight holes.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape being covered (before padding).
    patch_size : sequence of int
        Patch shape, as returned by :func:`plan_grid`.
    starts : sequence of sequence of int
        Start offsets per axis, as returned by :func:`plan_grid`.
    max_margin : int, optional
        Longest ramp, in pixels. The actual ramp is the smaller of this and the
        real overlap, so a narrow overlap simply gets a shorter ramp.
        Default is 16.
    max_cache_bytes : int, optional
        Ceiling on cached tile weights. Grids with many large 3D tiles would
        otherwise cache more than the result itself; past the ceiling weights are
        recomputed on demand. Default is 256 MB.
    """

    def __init__(self, shape, patch_size, starts, max_margin: int = 16,
                 max_cache_bytes: int = 256 << 20):
        self.patch_size = tuple(int(p) for p in patch_size)
        self.starts = [list(map(int, s)) for s in starts]
        self.canvas_shape = tuple(max(p, int(s)) for p, s in zip(self.patch_size, shape))
        self._profiles = [_axis_profiles(p, s, max_margin)
                          for p, s in zip(self.patch_size, self.starts)]
        self._cache = {}
        self._cache_bytes = 0
        self._max_cache_bytes = max_cache_bytes

        coverage = np.zeros(self.canvas_shape, dtype=np.float32)
        for index in np.ndindex(*(len(s) for s in self.starts)):
            coverage[self.region(index)] += self._raw(index)
        # Every pixel is covered by construction (starts span the axis), but guard
        # against a degenerate plan rather than dividing by zero.
        np.maximum(coverage, np.finfo(np.float32).tiny, out=coverage)
        self._coverage = coverage

    def region(self, index) -> tuple:
        """Slices placing tile ``index`` (a per-axis tuple) on the canvas."""
        return tuple(slice(s[i], s[i] + p)
                     for i, s, p in zip(index, self.starts, self.patch_size))

    def _raw(self, index) -> np.ndarray:
        w = self._profiles[0][index[0]]
        for axis in range(1, len(index)):
            w = w[..., None] * self._profiles[axis][index[axis]]
        return w

    def weight(self, index) -> np.ndarray:
        """Normalised weight of tile ``index``, shaped like the patch."""
        key = tuple(index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        w = self._raw(key) / self._coverage[self.region(key)]
        if self._cache_bytes + w.nbytes <= self._max_cache_bytes:
            self._cache[key] = w
            self._cache_bytes += w.nbytes
        return w


def redundancy(shape, patch_size, starts) -> float:
    """Pixels the grid feeds to the network, divided by pixels in the image.

    1.0 means no pixel is predicted twice; 2.0 means the network does twice the
    necessary work. Useful for asserting that a tiling change actually helped.
    """
    processed = 1
    for patch, st in zip(patch_size, starts):
        processed *= patch * len(st)
    return processed / float(np.prod(shape))
