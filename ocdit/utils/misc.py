import numpy as np
import torch
from torch.distributions import normal

_constant_cache = dict()


def truncated_exp_normal_dist(
    mean: float, std: float, min: float, max: float, size: int, device: torch.device
):
    dist = normal.Normal(
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    )
    sigma = (dist.sample([size]) * std + mean).exp()
    for i in range(size):
        while sigma[i] < min or sigma[i] > max:
            sigma[i] = (dist.sample() * std + mean).exp()
    return sigma


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )
