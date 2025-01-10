from typing import Optional, Tuple, Union

import torch as th
from numpy.typing import ArrayLike

from dexp_dl.transforms.basetransform import BaseTransform
from dexp_dl.transforms.clicker import Clicker
from dexp_dl.transforms.utils import find_object


class ZoomIn(BaseTransform):
    def __init__(
        self,
        expansion_rate: float = 1.4,
        min_size: int = 32,
        mask: Optional[th.Tensor] = None,
        device: Union[int, str] = "cpu",
        logits: bool = True,
    ):
        self.expansion_rate = expansion_rate
        self.half_min_size = min_size / 2
        self._slicing: Optional[Tuple[slice]] = None
        self._mask: Optional[ArrayLike] = None
        self._logits = logits
        self.device = device
        if mask is not None:
            self.fit(mask)

    def reset(self) -> None:
        self._mask = None
        self._slicing = None

    @staticmethod
    def include_positive_clicks(
        slicing: Tuple[slice], clicker: Clicker
    ) -> Tuple[slice]:
        return tuple(
            slice(min(m, s.start), max(M, s.stop))
            for s, m, M in zip(slicing, clicker.min_position, clicker.max_position)
        )

    def expand_slicing(self, slicing: Tuple[slice]) -> Tuple[slice]:
        new_slicing = []
        for b, s in zip(slicing, self._mask.shape):
            half_len = (b.stop - b.start) * 0.5 * self.expansion_rate
            if half_len < self.half_min_size:
                half_len = self.half_min_size

            mid = (b.start + b.stop) * 0.5
            lower = max(0, int(round(mid - half_len)))
            upper = min(s, int(round(mid + half_len)))
            new_slicing.append(slice(lower, upper))
        return tuple(new_slicing)

    def fit(self, pred: th.Tensor, clicker: Optional[Clicker] = None) -> None:
        pred = (pred >= 0.5).squeeze()
        if pred.sum() == 0:
            return
        self._mask = pred
        slicing = find_object(self._mask)
        if clicker is not None:
            slicing = self.include_positive_clicks(slicing, clicker)
        self._slicing = self.expand_slicing(slicing)

    def _get_slicing(self, ndim: int) -> Tuple[slice]:
        feature_dims = ndim - len(self._slicing)
        return (slice(None),) * feature_dims + self._slicing

    def transform(self, image: th.Tensor) -> th.Tensor:
        if self._slicing is None:
            return image
        slicing = self._get_slicing(image.ndim)
        return image[slicing]

    def inverse(self, crop: th.Tensor, logits: Optional[bool] = None) -> th.Tensor:
        if self._slicing is None:
            return crop

        if logits is None:
            logits = self._logits

        shape = (1,) * (crop.ndim - self._mask.ndim) + self._mask.shape
        image = th.full(
            shape, 0 if logits else -100, dtype=crop.dtype, device=self.device
        )
        slicing = self._get_slicing(image.ndim)
        image[slicing] = crop
        return image
