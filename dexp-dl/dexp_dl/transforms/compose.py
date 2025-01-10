from typing import List

import torch as th

from dexp_dl.transforms.basetransform import BaseTransform


class Compose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def transform(self, tensor: th.Tensor) -> th.Tensor:
        for tr in self.transforms:
            tensor = tr.transform(tensor)
        return tensor

    def inverse(self, tensor: th.Tensor) -> th.Tensor:
        for tr in reversed(self.transforms):
            tensor = tr.inverse(tensor)
        return tensor


class BatchCompose(BaseTransform):
    def __init__(self, transforms: List[Compose]):
        self.transforms = transforms

    def transform(self, tensors: th.Tensor) -> th.Tensor:
        assert len(tensors) == len(self.transforms)
        batch = [c.transform(x) for c, x in zip(self.transforms, tensors)]
        return th.stack(batch)

    def inverse(self, tensors: th.Tensor) -> th.Tensor:
        assert len(tensors) == len(self.transforms)
        batch = [c.inverse(x) for c, x in zip(self.transforms, tensors)]
        return th.stack(batch)
