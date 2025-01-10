from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import torch as th
from numpy.typing import ArrayLike
from tiler import Merger
from torch.utils.data import IterableDataset


def to_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def take(array: ArrayLike, indices: Union[int, List[int]], axis: int) -> ArrayLike:

    if isinstance(array, np.ndarray):
        return np.take(array, indices, axis)
    elif isinstance(array, th.Tensor):
        sliced = np.take(array.numpy(), indices, axis)
        return th.Tensor(sliced)
    else:
        raise NotImplementedError


def collate_any_fn(sequence: Any) -> Any:
    n_items = len(sequence[0])
    batch = [[] for _ in range(n_items)]
    for items in sequence:
        for i, item in enumerate(items):
            batch[i].append(item)

    for i in range(n_items):
        if isinstance(batch[i][0], th.Tensor) or isinstance(batch[i][0], np.ndarray):
            batch[i] = th.stack(batch[i], dim=0)
    return batch


class SequentialMerger(Merger):
    """
    Modified from tiler.Merger.
    tiler package version is fixed to avoid bugs.
    """

    def __init__(self, dtype: np.dtype, *args, **kwargs):
        self._dtype = dtype
        super().__init__(*args, **kwargs)
        self._merge_count = 0
        del self.data_visits
        del self.weights_sum

    def add(self, data: ArrayLike) -> Optional[ArrayLike]:
        self._add(self._merge_count, data)
        self._merge_count += 1
        if self._merge_count < self.tiler.n_tiles:
            return None
        array = self.merge()
        self._merge_count = 0
        self.reset()
        return array

    def reset(self) -> None:
        """Reset data and normalization buffers.

        Should be done after finishing merging full tile set and before starting processing the next tile set.

        Returns:
            None
        """

        padded_data_shape = self.tiler._new_shape

        # Image holds sum of all processed tiles multiplied by the window
        if self.logits:
            self.data = np.zeros((self.logits, *padded_data_shape), dtype=self._dtype)
        else:
            self.data = np.zeros(padded_data_shape, dtype=self._dtype)

    def _add(self, tile_id: int, data: np.ndarray) -> None:
        """Adds `tile_id`-th tile into Merger.

        Args:
            tile_id (int): Specifies which tile it is.
            data (np.ndarray): Specifies tile data.

        Returns:
            None
        """
        if tile_id < 0 or tile_id >= len(self.tiler):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}. "
                f"There are {len(self.tiler)} tiles, starting from index 0."
            )

        data_shape = np.array(data.shape)
        expected_tile_shape = (
            ((self.logits,) + tuple(self.tiler.tile_shape))
            if self.logits > 0
            else tuple(self.tiler.tile_shape)
        )

        if self.tiler.mode != "irregular":
            if not np.all(np.equal(data_shape, expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"does not fit expected tile shape ({expected_tile_shape})."
                )
        else:
            if not np.all(np.less_equal(data_shape, expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"must be less or equal than tile shape ({expected_tile_shape})."
                )

        # Select coordinates for data
        shape_diff = expected_tile_shape - data_shape
        a, b = self.tiler.get_tile_bbox_position(tile_id, with_channel_dim=True)

        sl = [slice(x, y - shape_diff[i]) for i, (x, y) in enumerate(zip(a, b))]
        win_sl = [
            slice(None, -diff) if (diff > 0) else slice(None, None)
            for diff in shape_diff
        ]

        # TODO check for self.data and data dtypes mismatch?
        if self.logits > 0:
            self.data[tuple([slice(None, None, None)] + sl)] = (
                data * self.window[tuple(win_sl[1:])]
            )
        else:
            self.data[tuple(sl)] = data * self.window[tuple(win_sl)]


class ConcatIterableDataset(IterableDataset):
    def __init__(self, datasets: List[IterableDataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self) -> Iterator:
        for items in zip(*self.datasets):
            yield from items

    def __len__(self) -> int:
        return min(len(d) for d in self.datasets) * len(self.datasets)
