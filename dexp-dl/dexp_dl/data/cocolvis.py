import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from dexp_dl.transforms.clicker import Click, get_num_of_clicks, random_clicks


class CocoLvisDataset(Dataset):
    # combined annotations from: https://github.com/saic-vul/ritm_interactive_segmentation
    # images from v1.0: https://www.lvisdataset.org/dataset
    def __init__(
        self,
        dataset_path: Union[str, Path],
        split: str,
        stuff_prob: float = 0.0,
        anno_file: str = "hannotation.pickle",
        min_object_area: int = 80,
        max_num_clicks: int = 12,
        transforms: Optional[Callable] = None,
    ):

        self.transforms = transforms
        self._path = (
            dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
        )
        self._path = self._path / split

        self._min_object_area = min_object_area
        self._max_num_clicks = max_num_clicks

        self._images_path = self._path / "images"
        self._masks_path = self._path / "masks"
        self.stuff_prob = stuff_prob

        with open(self._path / anno_file, "rb") as f:
            self.dataset_samples = sorted(pickle.load(f).items())

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike, List[Click]]:
        """
        return: image, mask, points
        """
        assert isinstance(index, int)
        image_id, sample = self.dataset_samples[index]
        im_path = self._images_path / f"{image_id}.jpg"

        image = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f"{image_id}.pickle"
        with open(packed_masks_path, "rb") as f:
            encoded_layers, objs_mapping = pickle.load(f)

        layers = [cv2.imdecode(l, cv2.IMREAD_UNCHANGED) for l in encoded_layers]
        layers = np.stack(layers, axis=-1)

        instances_info = deepcopy(sample["hierarchy"])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {"children": [], "parent": None, "node_level": 0}
                instances_info[inst_id] = inst_info
            inst_info["mapping"] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and torch.rand(1) < self.stuff_prob:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                instances_info[inst_id] = {
                    "mapping": objs_mapping[inst_id],
                    "parent": None,
                    "children": [],
                }
        else:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        selected_object_area = 0
        while selected_object_area < self._min_object_area and len(instances_info) > 0:
            keys = list(instances_info.keys())
            rnd_index = keys[torch.randint(len(keys), (1,)).item()]
            layer_index, label = instances_info[rnd_index]["mapping"]
            mask = (layers[..., layer_index] == label).astype(np.uint8)
            selected_object_area = mask.sum()
            del instances_info[rnd_index]

        if len(instances_info) == 0:
            # skip to next image if no object larger than min object area found
            return self[(index + 1) % len(self)]

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        max_positive_clicks, max_negative_clicks = get_num_of_clicks(
            self._max_num_clicks
        )

        return (
            image,
            mask,
            random_clicks(
                mask.squeeze().numpy(), max_positive_clicks, max_negative_clicks
            ),
        )
