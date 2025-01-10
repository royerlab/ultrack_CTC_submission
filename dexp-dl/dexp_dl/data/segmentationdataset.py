from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from dexp_dl.data.utils import to_path


class FileDataset(Dataset):
    def __init__(
        self,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        file_key: str,
        loader: Callable,
        transforms: Optional[Callable] = None,
    ):

        self.images_dir = to_path(images_dir)
        self.labels_dir = to_path(labels_dir)
        self.file_key = file_key
        self.loader = loader
        self.transforms = transforms

        self.images_path = list(self.images_dir.glob(self.file_key))
        self.labels_path = [self.labels_dir / p.name for p in self.images_path]

        for p in self.labels_path:
            if not p.exists():
                raise ValueError(f"Label image {str(p)} not found.")

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike]:
        im = self.loader(str(self.images_path[index])).astype(np.float32)
        lb = self.loader(str(self.labels_path[index])).astype(np.int32)

        if self.transforms is not None:
            im, lb = self.transforms(im, lb)

        return im, lb
