import torch as th


class BaseTransform:
    def transform(self, tensor: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def inverse(self, tensor: th.Tensor) -> th.Tensor:
        raise NotImplementedError
