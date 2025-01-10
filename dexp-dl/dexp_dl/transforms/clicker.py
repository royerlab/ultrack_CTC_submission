from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch as th
from numpy.typing import ArrayLike
from scipy import ndimage as ndi
from skimage.morphology import ball, binary_dilation, binary_erosion, disk


@dataclass
class Click:
    position: Tuple[int]
    is_positive: bool

    def fill_image(
        self, image: ArrayLike, sphere_coords: Tuple[ArrayLike], radius: int
    ):
        map = image[int(self.is_positive)]
        coords = tuple(
            np.minimum(np.maximum(0, c + p - radius), s - 1)
            for c, p, s in zip(sphere_coords, self.position, map.shape)
        )
        map[coords] = 1


class Clicker:
    def __init__(
        self,
        radius: int,
        target: Optional[th.Tensor] = None,
        shape: Optional[th.Size] = None,
        device: Union[int, str] = "cpu",
    ):

        self._target = target >= 0.5 if target is not None else target
        self._shape = self._target.shape if shape is None else shape

        if target is None and shape is None:
            raise ValueError("`target` segmentation or `shape` must be provided.")

        if self._target is not None and self._shape != self._target.shape:
            raise ValueError("`target.shape` and `shape` must be equal.")

        self._radius = radius
        self.device = device
        if len(self._shape) == 2:
            self._sphere_coords = np.where(disk(radius))
        elif len(self._shape) == 3:
            self._sphere_coords = np.where(ball(radius))
        else:
            raise NotImplementedError

        self.min_position: List[int] = [1000000] * len(self._shape)
        self.max_position: List[int] = [-1] * len(self._shape)

        self.clicks: List[Click] = []

    @staticmethod
    def get_largest_error_points(
        pred_mask: th.Tensor, gt_mask: th.Tensor
    ) -> Tuple[Tuple[int], float]:
        errors = th.logical_xor(gt_mask, pred_mask)
        errors *= gt_mask
        if errors.sum() == 0:
            return None, 0
        errors = errors.cpu().numpy().astype(bool)

        edt = ndi.distance_transform_edt(errors)
        peak = ndi.maximum_position(edt)
        return peak, edt[peak].item()

    def _find_click(self, pred: th.Tensor) -> Optional[Click]:
        fg_click, fg_dist = self.get_largest_error_points(pred >= 0.5, self._target)
        bg_click, bg_dist = self.get_largest_error_points(
            pred < 0.5, th.logical_not(self._target)
        )

        if fg_click is None and bg_click is None:
            # no click found
            return None

        if fg_dist >= bg_dist:
            return Click(fg_click, True)
        else:
            return Click(bg_click, False)

    def add_click(self, click: Click) -> None:
        self.clicks.append(click)
        if click.is_positive:
            for i, p in enumerate(click.position):
                self.min_position[i] = min(self.min_position[i], p)
                self.max_position[i] = max(self.max_position[i], p)

    def add_clicks(self, clicks: Click) -> None:
        for click in clicks:
            self.add_click(click)

    def query_next_map(
        self,
        pred: Optional[th.Tensor] = None,
        add_click: bool = True,
        dtype: th.dtype = th.float32,
    ) -> th.Tensor:

        if pred is None:
            pred = th.zeros(size=self._shape, dtype=dtype, device=self.device)
        else:
            pred = pred.squeeze(0)

        assert pred.shape == self._shape, f"{pred.shape} and {self._shape}"

        if add_click:
            if self._target is None:
                raise ValueError(
                    "`target` must be provided to automatically add a click."
                )
            click = self._find_click(pred)
            if click is not None:
                self.add_click(click)

        maps = th.zeros((3, *self._shape), dtype=dtype).numpy()
        for c in self.clicks:
            c.fill_image(maps, self._sphere_coords, self._radius)

        maps[2, ...] = pred.cpu().numpy() >= 0.5
        return th.tensor(maps, dtype=dtype, device=self.device)

    def clear(self) -> None:
        self.clicks.clear()


def get_clicks_from_mask(
    mask: ArrayLike, n_clicks: int, is_positive: bool
) -> List[Click]:
    if n_clicks == 0 or mask.sum() == 0:
        return []
    positions = np.array(np.where(mask))
    indices = th.randint(positions.shape[1], size=(n_clicks,))
    return [Click(tuple(positions[:, i]), is_positive) for i in indices]


def get_disk(ndim: int, radius: int) -> ArrayLike:
    if ndim == 2:
        return disk(radius)
    elif ndim == 3:
        return ball(radius)
    else:
        raise NotImplementedError


def get_num_of_clicks(max_num_clicks: int) -> Tuple[int, int]:
    max_negative_clicks = th.randint(0, max_num_clicks, (1,)).item()
    max_positive_clicks = max_num_clicks - max_negative_clicks + 1
    return max_positive_clicks, max_negative_clicks


def random_clicks(
    mask: ArrayLike,
    max_positives: int,
    max_negatives: int,
    min_area_object_prop: float = 0.5,
    background_dilation_radius: int = 15,
    max_object_erosion: int = 10,
) -> List[Click]:

    mask = mask.astype(bool)
    area = mask.sum()

    obj_elem = get_disk(mask.ndim, 1)
    obj_mask = mask.copy()
    for i in range(max_object_erosion):
        tmp_mask = binary_erosion(obj_mask, obj_elem)
        if tmp_mask.sum() <= min_area_object_prop * area:
            break
        obj_mask = tmp_mask

    bkg_mask = binary_dilation(mask, get_disk(mask.ndim, background_dilation_radius))
    bkg_mask[mask] = 0

    n_positives = th.randint(1, max_positives + 1, (1,)).item()
    n_negatives = th.randint(0, max_negatives + 1, (1,)).item()

    return get_clicks_from_mask(obj_mask, n_positives, True) + get_clicks_from_mask(
        bkg_mask, n_negatives, False
    )
