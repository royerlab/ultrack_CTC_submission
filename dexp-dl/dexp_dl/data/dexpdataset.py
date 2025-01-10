from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile as tiff
import tiler
import torch as th
import zarr
from dexp.datasets.zarr_dataset import ZDataset
from numpy.typing import ArrayLike
from scipy.ndimage import find_objects
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

from dexp_dl.data.utils import SequentialMerger
from dexp_dl.transforms.utils import random_slice


class DexpQueueDataset(IterableDataset):
    def __init__(
        self,
        dataset: ZDataset,
        channels: Union[Sequence[str], str],
        num_iterations: int,
        patch_size: Tuple[int],
        validation_fun: Callable,
        transforms: Optional[Callable] = None,
        queue_size: int = 4,
        queue_iterations: int = 20,
    ):
        super().__init__()

        self.dataset = dataset
        self.channels = [channels] if isinstance(channels, str) else channels
        self.validation_fun = validation_fun
        self.transforms = transforms

        self._num_iterations = num_iterations
        self._patch_size = patch_size
        self._queue_size = queue_size
        self._queue_iterations = queue_iterations

        for ch in self.channels:
            if ch not in self.dataset.channels():
                raise ValueError(f"Channel {ch} not found in dataset.")

        self.shape = self.dataset.shape(self.channels[0])
        for ch in self.channels[1:]:
            if self.shape != self.dataset.shape(ch):
                raise NotImplementedError(
                    "DexpDataset does not support channels with different shapes yet."
                )

        if len(self._patch_size) != len(self.shape) - 1:
            raise ValueError(
                f"Patch dimensions {self._patch_size} must be one less than the array shape {self.shape}."
            )

        # FIXME print("Don't forget to implement the worker_init_fn so they have different rng states")

    def __len__(self) -> int:
        return self._num_iterations

    def __iter__(self) -> Iterator:
        for _ in range(self._num_iterations // self._queue_iterations):
            queue = self._load_queue()
            count = 0
            while count < self._queue_iterations:
                patch = self._sample_patch(queue)
                if self.validation_fun(patch):
                    if self.transforms is not None:
                        patch = self.transforms(patch)
                    count += 1
                    yield patch

    def _sample_patch(self, queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        slicing = random_slice(self.shape[1:], self._patch_size)
        index = th.randint(len(queue), size=(1,)).item()
        stack = {}
        for ch in self.channels:
            stack[ch] = queue[index][ch][slicing]
        return stack

    def _load_queue(self) -> List[Dict[str, Any]]:
        queue = []
        random_time_pts = th.randint(self.shape[0], size=(self._queue_size,)).numpy()

        for t in random_time_pts:
            stack = {
                ch: np.asarray(self.dataset.get_stack(ch, t)) for ch in self.channels
            }
            queue.append(stack)

        return queue


class DexpTileDataset(IterableDataset):
    def __init__(
        self,
        array: zarr.Array,
        tile_shape: Union[int, Tuple[int]],
        overlap: Union[int, Tuple[int]] = 0,
        transforms: Optional[Callable] = None,
        starting_index: int = 0,
        rank: int = 0,
        world_size: int = 1,
        flip_odd_index: bool = False,
    ) -> None:

        super().__init__()

        self._starting_index = starting_index
        self._world_size = world_size
        self._rank = rank
        self._array = array
        self.transforms = transforms
        self._flip_odd_index = flip_odd_index
        self._is_flipped = False

        if isinstance(tile_shape, int):
            tile_shape = [tile_shape for _ in range(3)]
        else:
            tile_shape = list(tile_shape)

        data_shape = self._array.shape[1:]
        if isinstance(overlap, int):
            overlap = [overlap, overlap, overlap]
        else:
            overlap = list(overlap)

        for i in range(3):
            tile_shape[i] = min(tile_shape[i], data_shape[i])
            if data_shape[i] == tile_shape[i]:
                overlap[i] = 0

        self._tiler = tiler.Tiler(
            data_shape=data_shape,
            tile_shape=tile_shape,
            overlap=overlap,
            mode="reflect",
        )

        print(f"Stack shape {data_shape}")
        print("Using tiles with:")
        print(f"Size: {tile_shape}")
        print(f"Overlap: {overlap}")

        self._mergers: Dict[str, SequentialMerger] = {}

    @property
    def n_tiles(self) -> int:
        return self._tiler.n_tiles

    @property
    def n_time_points(self) -> int:
        quo = (self._array.shape[0] - self._starting_index) // self._world_size
        rem = (self._array.shape[0] - self._starting_index) % self._world_size
        return quo + int(self._rank < rem)

    def __len__(self) -> int:
        return self.n_time_points * self.n_tiles

    def __iter__(self) -> Iterator:
        for t in range(
            self._starting_index + self._rank, self._array.shape[0], self._world_size
        ):
            array = self._array[t]

            self._is_flipped = self._flip_odd_index and t % 2 == 1

            if self._is_flipped:
                array = np.flip(array)

            for _, tile in self._tiler(array, copy_data=False):
                if self.transforms is not None:
                    tile = self.transforms(tile)
                yield t, tile

    def write_tile(self, array: ArrayLike, key: str = "") -> Optional[ArrayLike]:
        if key not in self._mergers:
            self._mergers[key] = SequentialMerger(array.dtype, self._tiler)

        maybe_array = self._mergers[key].add(array)
        if maybe_array is not None and self._is_flipped:
            maybe_array = np.flip(maybe_array)

        return maybe_array


def _add_margin(slicing: Tuple[slice], shape: Tuple[int], margin: int) -> Tuple[slice]:
    return tuple(
        slice(max(0, s.start - margin), min(d, s.stop + margin))
        for s, d in zip(slicing, shape)
    )


def _change_ratio(slicing: Tuple[slice], ratio: int) -> Tuple[slice]:
    return tuple(slice(s.start // ratio, s.end // ratio) for s in slicing)


def dexp_dataset_to_instances(
    dataset: ZDataset,
    out_dir: Union[str, Path],
    im_channel: str,
    lb_channel: str,
    lb_im_z_ratio: int = 1,
    margin: int = 24,
) -> None:

    out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
    lb_dir = out_dir / "labels"
    im_dir = out_dir / "images"
    lb_dir.mkdir(parents=True, exist_ok=True)
    im_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for t in tqdm(range(dataset.nb_timepoints(lb_channel)), "Exporting instances"):
        label = dataset.get_stack(lb_channel, t)
        image = dataset.get_stack(im_channel, t)
        objects = find_objects(label)
        for lb, slicing in enumerate(objects):
            if slicing is None:
                # necessary ue to find_objects behavior
                continue
            slicing = _add_margin(slicing, label.shape, margin)
            inst_label = (label[slicing] == lb + 1).astype(np.uint8)
            if lb_im_z_ratio != 1:
                slicing = _change_ratio(slicing, lb_im_z_ratio)
            inst_image = image[slicing]

            name = f"{count:08}.tif"
            tiff.imwrite(str(lb_dir / name), inst_label)
            tiff.imwrite(str(im_dir / name), inst_image)
            count += 1
