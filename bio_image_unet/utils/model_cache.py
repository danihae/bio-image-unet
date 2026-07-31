"""Process-level cache for loaded model weights.

Batch workflows construct a ``Predict`` object per file, and each construction
otherwise repeats ``torch.load`` + ``load_state_dict`` + a host-to-device copy of
the same checkpoint. Caching the built model keeps repeated predictions on one
checkpoint from paying that cost again.

The cache holds one entry per (checkpoint, mtime, device, precision, network,
extra kwargs). It is deliberately tiny: models are large, and workflows almost
always reuse a single checkpoint.
"""
import os
from collections import OrderedDict

import torch

__all__ = ['load_cached_model', 'clear_model_cache']

_CACHE = OrderedDict()
_MAX_ENTRIES = 2


def clear_model_cache():
    """Drop every cached model (e.g. to release GPU memory)."""
    _CACHE.clear()


def _load_params(path, device):
    # These checkpoints pickle more than tensors (optimizer state, config dicts),
    # so weights_only=False is required and stated rather than left to the default.
    return torch.load(path, map_location=device, weights_only=False)


def load_cached_model(model_params, network, device, precision, use_cache=True,
                      from_params=None, **network_kwargs):
    """Build (or fetch) an eval-mode model and return it with its checkpoint dict.

    Parameters
    ----------
    model_params : str or dict
        Path to a ``.pt`` checkpoint, or an already-loaded checkpoint dict (never cached).
    network : type
        Network class to instantiate.
    device : torch.device
        Device to place the model on.
    precision : torch.dtype
        Weight dtype; the model is cast to it after loading.
    use_cache : bool, optional
        Set False to bypass the cache entirely. Default is True.
    from_params : dict, optional
        Constructor arguments to read out of the checkpoint, mapping argument name
        to the fallback used when the checkpoint does not carry it.
    **network_kwargs
        Further keyword arguments passed straight to the network constructor. These
        take part in the cache key, so two callers asking for different variants of
        the same checkpoint get different models.

    Returns
    -------
    model : torch.nn.Module
        The model, in eval mode, on ``device`` with dtype ``precision``.
    params : dict
        The checkpoint dictionary.
    """
    from_params = dict(from_params or {})

    if isinstance(model_params, dict):
        params = model_params
        return _build(params, network, device, precision, from_params, network_kwargs), params

    key = None
    if use_cache:
        try:
            stamp = os.path.getmtime(model_params)
        except OSError:
            stamp = None
        key = (os.path.abspath(model_params), stamp, str(device), str(precision),
               network.__name__, tuple(sorted(network_kwargs.items())),
               tuple(sorted(from_params.items())))
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    params = _load_params(model_params, device)
    model = _build(params, network, device, precision, from_params, network_kwargs)

    if use_cache:
        _CACHE[key] = (model, params)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return model, params


def _build(params, network, device, precision, from_params, network_kwargs):
    kwargs = {name: params.get(name, default) for name, default in from_params.items()}
    kwargs.update(network_kwargs)
    model = network(in_channels=params['in_channels'],
                    n_filter=params['n_filter'],
                    output_heads=params['output_heads'],
                    **kwargs).to(device)
    model.load_state_dict(params['state_dict'])
    model.to(precision)
    model.eval()
    return model
