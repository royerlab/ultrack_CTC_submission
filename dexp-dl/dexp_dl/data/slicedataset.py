from typing import Callable, Optional, Tuple

from numpy.typing import ArrayLike

from dexp_dl.data.segmentationdataset import FileDataset
from dexp_dl.data.utils import take


class SliceDataset(FileDataset):
    # TODO: there should be a function to split it into two datasets with disjoints original images
    def __init__(
        self,
        slice_thickness: int,
        after_transforms: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.slice_thickness = slice_thickness
        self.after_transforms = after_transforms
        assert self.slice_thickness % 2 == 1, "Slices thickness must be an odd number."
        self.length, self.depth = self._validate_images_and_length()

    def __len__(self) -> int:
        return self.length

    def _validate_images_and_length(self) -> Tuple[int, int]:
        # it assumes all images have the same length
        image = self.loader(self.images_path[0])
        if image.ndim != 3:
            raise RuntimeError(f"Images must 3 --- {image.ndim} found.")

        return image.shape[-3] * len(self.images_path), image.shape[-3]

    def _get_slice(self, image: ArrayLike, index: int) -> ArrayLike:
        d = self.slice_thickness // 2
        indices = [
            min(max(i, 0), len(image) - 1) for i in range(index - d, index + d + 1)
        ]
        return image[indices]

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike]:
        im_index = index // self.depth
        slice_index = index % self.depth
        im, lb = super().__getitem__(im_index)
        im, lb = self._get_slice(im, slice_index), take(lb, slice_index, -3)
        if self.after_transforms:
            im, lb = self.after_transforms(im, lb)
        return im, lb
