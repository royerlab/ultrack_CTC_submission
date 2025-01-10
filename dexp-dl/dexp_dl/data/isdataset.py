from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch as th
from torch.utils.data import Dataset

from dexp_dl.data.segmentationdataset import FileDataset
from dexp_dl.transforms.clicker import Click, get_num_of_clicks, random_clicks


class ISTileDataset(FileDataset):
    def __init__(
        self,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        file_key: str,
        loader: Callable,
        max_num_clicks: int,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            images_dir=images_dir,
            labels_dir=labels_dir,
            file_key=file_key,
            loader=loader,
            transforms=transforms,
        )
        self._max_num_clicks = max_num_clicks

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor, List[Click]]:
        image, mask = super().__getitem__(index)

        max_pos_clicks, max_neg_clicks = get_num_of_clicks(self._max_num_clicks)

        clicks = random_clicks(
            mask.squeeze().cpu().numpy(), max_pos_clicks, max_neg_clicks
        )
        return image, mask, clicks


class ISOnlineDataset(Dataset):
    def __init__(
        self, max_num_clicks: int, transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self._max_num_clicks = max_num_clicks
        self.transforms = transforms
        self._images = []
        self._mask = []

    def append(self, image: th.Tensor, mask: th.Tensor) -> None:
        assert image.ndim == mask.ndim
        assert image.shape[-1:] == mask.shape[-1:]
        self._images.append(image)
        self._mask.append(mask)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor, List[Click]]:
        index = index % len(self._images)

        image, mask = self._images[index], self._mask[index]

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        max_pos_clicks, max_neg_clicks = get_num_of_clicks(self._max_num_clicks)

        clicks = random_clicks(
            mask.squeeze().cpu().numpy(), max_pos_clicks, max_neg_clicks
        )
        return image, mask, clicks
