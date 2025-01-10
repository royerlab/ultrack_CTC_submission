from functools import partial
from typing import Callable, Optional, Tuple

import torch as th
import torch.nn.functional as F

from dexp_dl.transforms.basetransform import BaseTransform


class Resize(BaseTransform):
    def __init__(self, target_shape: Tuple[int]) -> None:
        assert (
            len(target_shape) == 3 or len(target_shape) == 2
        ), f"Length {len(target_shape)} found."
        self._forward: Callable = ...
        self._backward: Optional[Callable] = None
        self._mode: str = ...
        self._target_shape: Tuple[int] = ...
        self.target_shape = target_shape

    @property
    def target_shape(self) -> Tuple[int]:
        return self._target_shape

    @target_shape.setter
    def target_shape(self, shape: Tuple[int]) -> None:
        if len(shape) == 2:
            self._mode = "bilinear"
        elif len(shape) == 3:
            self._mode = "trilinear"
        else:
            raise NotImplementedError
        self._target_shape = shape
        self._forward = partial(
            F.interpolate, size=shape, mode=self._mode, align_corners=True
        )

    def transform(self, image: th.Tensor) -> th.Tensor:
        self._backward = partial(
            F.interpolate,
            mode=self._mode,
            align_corners=True,
            size=image.shape[-len(self._target_shape) :],
        )
        return self._forward(image.unsqueeze(0)).squeeze(0)

    def inverse(self, image: th.Tensor) -> th.Tensor:
        if self._backward is None:
            raise RuntimeError("Resize transform must be executed before inverse.")
        return self._backward(image.unsqueeze(0)).squeeze(0)
