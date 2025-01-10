import torch as th

from dexp_dl.models.unet import UNet


def test_2d_unet_forward():
    shapes = [(1, 3, 367, 477), (1, 3, 512, 512)]
    for shape in shapes:
        x = th.zeros(shape)
        net = UNet(3, 1)
        y = net.forward(x)
        assert y.shape[2:] == shape[2:]


def test_3d_unet_forward():
    shapes = [(1, 1, 47, 367, 477), (1, 1, 64, 256, 256)]
    for shape in shapes:
        x = th.zeros(shape)
        net = UNet(1, 1, conv_layer=th.nn.Conv3d)
        y = net.forward(x)
        assert y.shape[2:] == shape[2:]
