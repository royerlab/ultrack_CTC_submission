from typing import Tuple

import numpy as np
import torch as th
from numpy.typing import ArrayLike


def random_slice(orig_shape: Tuple[int], crop_shape: Tuple[int]) -> Tuple[slice]:
    assert len(orig_shape) == len(crop_shape)

    start = tuple(
        0 if i_size - c_size <= 0 else th.randint(i_size - c_size, (1,)).item()
        for i_size, c_size in zip(orig_shape, crop_shape)
    )
    return tuple(slice(s, s + c_size) for s, c_size in zip(start, crop_shape))


def find_object(mask: th.Tensor) -> Tuple[slice]:
    indices = th.nonzero(mask, as_tuple=True)
    return tuple(slice(i[0].item(), i[-1].item() + 1) for i in indices)


def pad(image: ArrayLike, shape: Tuple[int]) -> ArrayLike:
    dif = tuple(int(max(0, s - d)) for s, d in zip(shape, image.shape))
    dif = dif + (0,) * (image.ndim - len(shape))  # expanding if it contains add dims
    padding = tuple((s - s // 2, s // 2) for s in dif)
    return np.pad(image, padding)
