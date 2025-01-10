import torch as th

from dexp_dl.transforms import Resize


def test_resize():
    image = th.randn((3, 256, 512, 512), dtype=th.float32)
    target_shape = (400,) * 3
    resize = Resize(target_shape)

    resized = resize.transform(image)
    assert resized.shape[0] == image.shape[0] and resized.shape[1:] == target_shape

    inverse = resize.inverse(resized)
    assert inverse.shape == image.shape
