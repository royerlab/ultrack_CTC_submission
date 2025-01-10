from typing import Tuple, Union

import edt
import numpy as np
import scipy.ndimage as ndi
import torch as th
from numpy.typing import ArrayLike
from skimage import segmentation
from skimage.morphology import disk, ball, dilation

from dexp_dl.transforms.basetransform import BaseTransform
from dexp_dl.transforms.utils import pad, random_slice

__all__ = [
    "random_crop",
    "random_noise",
    "random_power",
    "random_slice",
    "random_transpose",
    "flip_axis",
    "upsample",
    "add_boundary",
    "add_edt",
    "dilate_edge_label",
    "gray_normalize",
    "rgb_normalize",
    "gray_to_rgb",
    "RGBNormalize",
]


def random_crop(
    image: ArrayLike, label: ArrayLike, crop_shape: Tuple[int], fill: bool = True
) -> Tuple[ArrayLike, ArrayLike]:
    assert label.ndim == len(crop_shape)
    slicing = random_slice(label.shape, crop_shape)
    image, label = image[slicing].copy(), label[slicing].copy()
    if fill and image.shape != crop_shape:
        image = pad(image, crop_shape)
        label = pad(label, crop_shape)
    return image, label


def random_noise(
    image: ArrayLike,
    label: ArrayLike,
    dist_support: Tuple[float, float] = (-0.05, 0.05),
) -> Tuple[ArrayLike, ArrayLike]:

    noise = th.rand(image.shape) * (dist_support[1] - dist_support[0]) + dist_support[0]
    image = image + noise.numpy()
    return image, label


def upsample(
    image: ArrayLike, label: ArrayLike, scale: Union[Tuple[float], float]
) -> Tuple[ArrayLike, ArrayLike]:
    assert not isinstance(scale, Tuple) or image.ndim == len(scale)
    image = ndi.zoom(image, zoom=scale, order=1)
    if label is not None:
        label = ndi.zoom(label, zoom=scale, order=0)
    return image, label


def add_boundary(image: ArrayLike, label: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    boundary = segmentation.find_boundaries(label, mode="outer")
    output = np.empty((2, *label.shape), dtype=np.int32)
    output[0, ...] = label != 0
    output[1, ...] = boundary
    return image, output


def _edt_inverse(
    edt: ArrayLike, mask: ArrayLike, upper_qtl: float = 0.999
) -> ArrayLike:
    edt = edt.astype(float)
    up = np.quantile(edt[mask], upper_qtl)
    edt = np.clip(edt, 0, up)
    edt = up - edt
    return edt / (edt.max() + 1e-8)


def add_edt(image: ArrayLike, label: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    binary_label = label != 0
    output = np.zeros((2, *label.shape), dtype=np.float32)
    output[0, ...] = binary_label
    if binary_label.sum() > 0:
        edt_map = edt.edt(label)
        output[1, ...] = _edt_inverse(edt_map, binary_label)
    return image, output


def gray_normalize(
    image: ArrayLike,
    label: ArrayLike,
    epsilon: float = 1e-8,
    lower: float = 0.005,
    upper: float = 0.9999,
) -> Tuple[ArrayLike, ArrayLike]:
    lb = np.quantile(image, lower)
    image = image - lb
    up = np.quantile(image, upper)
    image = np.clip(image, 0, up)
    image = image.astype(np.float32) / (up + epsilon)
    return image, label


def rgb_normalize(
    image: ArrayLike,
    label: ArrayLike,
    mean: Tuple[float] = (0.485, 0.456, 0.406),
    std: Tuple[float] = (0.229, 0.224, 0.225),
) -> Tuple[ArrayLike, ArrayLike]:
    assert image.dtype == np.uint8
    assert image.ndim == 3
    assert image.shape[2] == 3
    image = image / 255.0
    image -= np.array(mean)
    image /= np.array(std)
    return image, label


def flip_axis(
    image: ArrayLike, label: ArrayLike, axis: Union[Tuple[int], int], p: float = 0.5
) -> Tuple[ArrayLike, ArrayLike]:

    if isinstance(axis, int):
        axis = (axis,)

    for a in axis:
        if th.rand(1).item() < p:
            image = np.flip(image, a)
            if label is not None:
                label = np.flip(label, a).copy()

    return image.copy(), label


def random_transpose(
    image: ArrayLike, label: ArrayLike, p: float = 0.5
) -> Tuple[ArrayLike, ArrayLike]:
    assert image.ndim == label.ndim

    if th.rand(1).item() < p:
        axis = tuple(th.randperm(image.ndim).numpy())
        image = image.transpose(axis)
        label = label.transpose(axis)

    return image, label


def get_struct(dim: int, radius: int) -> ArrayLike:
    if dim == 2:
        return disk(radius)
    elif dim == 3:
        return ball(radius)
    raise NotImplementedError


def dilate_edge_label(
    image: ArrayLike, label: ArrayLike, radius: int,
) -> Tuple[ArrayLike, ArrayLike]:
    if radius == 0 or label is None:
        return image, label
    struct = get_struct(label.ndim - 1, radius=radius)
    label[1, ...] = dilation(label[1], struct)
    return image, label


def random_power(
    image: ArrayLike, labels: ArrayLike, range: Tuple[float] = (0.7, 1.3)
) -> Tuple[ArrayLike, ArrayLike]:
    gamma = th.distributions.Uniform(*range).sample().item()
    return np.power(image.astype(np.float32), gamma), labels


def gray_to_rgb(image: ArrayLike, labels: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    tiling = (3,) + image.ndim * (1,)
    return np.tile(image[np.newaxis, ...], tiling), labels


class RGBNormalize(BaseTransform):
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        scale: float = 255,
    ):
        self._mean = np.array(mean)
        self._std = np.array(std)
        self._scale = scale

    def transform(self, array: ArrayLike) -> ArrayLike:
        array = array / self._scale
        array -= self._mean
        array /= self._std
        return array

    def inverse(
        self, array: ArrayLike, is_CHW: bool = False, dtype: np.dtype = np.uint8
    ) -> ArrayLike:
        if is_CHW:
            ax = tuple(range(1, array.ndim)) + (0,)
            array = np.transpose(array, ax)
        array = array * self._std
        array += self._mean
        array *= self._scale
        if is_CHW:
            ax = (array.ndim - 1,) + tuple(range(array.ndim - 1))
            array = np.transpose(array, ax)
        return array.astype(dtype)
