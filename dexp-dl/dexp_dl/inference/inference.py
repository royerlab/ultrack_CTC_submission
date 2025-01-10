import logging
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch as th
from dexp.processing.utils.nd_slice import nd_split_slices, remove_margin_slice
from numpy.typing import ArrayLike

from dexp_dl.models.utils import load_weights

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(
        self,
        model: th.nn.Module,
        tile: Optional[Tuple] = None,
        margin: Union[int, Tuple] = 32,
        transforms: Optional[Callable] = None,
        after_transforms: Optional[Callable] = None,
        output_transforms: Optional[Callable] = None,
        num_outputs: int = 1,
        half_precision: bool = False,
        verbose: bool = False,
    ):

        self.model = model
        self.tile = tile
        self.margin = margin
        if self.tile is not None:
            if isinstance(self.margin, int):
                self.margin = (self.margin,) * len(self.tile)
            else:
                assert len(self.tile) == len(self.margin)

        self.transforms = transforms
        self.after_transforms = after_transforms
        self.output_transforms = output_transforms
        self.num_outputs = num_outputs
        self.half_precision = half_precision

        if verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    @staticmethod
    def _rescale_slice(slicing: Tuple[slice], scale: Tuple[int]) -> Tuple[slice]:
        return tuple(slice(s.start * c, s.stop * c) for s, c in zip(slicing, scale))

    def __call__(
        self,
        stack: ArrayLike,
        dtype: Optional[np.dtype] = np.float32,
        scale: Optional[Tuple[int]] = None,
        callback: Optional[Callable] = None,
    ) -> Any:

        if self.tile is None:
            preds = self._tile_predict(stack)
            if dtype is not None:
                preds = [p.astype(dtype) for p in preds]

            if callback is not None:
                callback()

        else:
            shape = stack.shape
            if scale is not None:
                shape *= np.array(scale)
            preds = [np.full(shape, -1, dtype=dtype) for _ in range(self.num_outputs)]

            slicing = zip(
                nd_split_slices(stack.shape, self.tile, self.margin),
                nd_split_slices(stack.shape, self.tile),
            )

            for src_slice, dst_slice in slicing:
                tile_slice = remove_margin_slice(stack.shape, src_slice, dst_slice)
                tile_pred = self._tile_predict(stack[src_slice])

                if scale is not None:
                    dst_slice = self._rescale_slice(dst_slice, scale)
                    tile_slice = self._rescale_slice(tile_slice, scale)

                for i in range(self.num_outputs):
                    preds[i][dst_slice] = tile_pred[i][tile_slice]

                if callback is not None:
                    callback()

        if self.num_outputs == 1:
            preds = preds[0]

        if self.output_transforms is not None:
            preds = self.output_transforms(preds)

        return preds

    def _tile_predict(self, stack: ArrayLike) -> ArrayLike:
        stack = np.asarray(stack)
        logger.info(f"Forwarding array (tile) of shape {stack.shape}")

        if self.transforms is not None:
            stack = self.transforms(stack)

        stack = stack.unsqueeze_(0)
        if th.cuda.is_available():
            stack = stack.cuda()

        self.model.eval()
        with th.no_grad():
            with th.cuda.amp.autocast() if self.half_precision else ExitStack():
                pred = self.model(stack)

        if self.after_transforms is not None:
            pred = self.after_transforms(pred)

        return pred.squeeze_().cpu().numpy()

    def n_tiles(self, shape: Tuple[int]) -> int:
        if self.tile is None:
            return 1
        return len(list(nd_split_slices(shape, self.tile, self.margin)))

    def load_weights(self, path: Union[str, Path]) -> None:
        load_weights(self.model, path)

        if th.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

    def cpu(self) -> "ModelInference":
        self.model = self.model.cpu()
        return self
