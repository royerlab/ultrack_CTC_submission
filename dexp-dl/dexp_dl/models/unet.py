from functools import partial
from typing import Callable, List, Tuple

import torch as th
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_layer: Callable):
        super().__init__()
        self.conv1 = conv_layer(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.conv2 = conv_layer(
            out_channels, out_channels, kernel_size=3, padding="same"
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class Encoder(nn.Module):
    def __init__(self, planes: Tuple[int], conv_layer: Callable, pool_layer: Callable):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(in_planes, out_planes, conv_layer)
                for in_planes, out_planes in zip(planes[:-1], planes[1:])
            ]
        )
        self.pool_layer = pool_layer

    def forward(self, x: th.Tensor) -> List[th.Tensor]:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
            x = self.pool_layer(x)
        return feats


class Decoder(nn.Module):
    def __init__(
        self, planes: Tuple[int], conv_layer: Callable, interp_layer: Callable
    ):
        super().__init__()
        self.first_block = conv_layer(
            planes[0], planes[1], kernel_size=3, padding="same"
        )
        self.blocks = nn.ModuleList(
            [
                Block(2 * in_planes, out_planes, conv_layer)
                for in_planes, out_planes in zip(planes[1:-1], planes[2:])
            ]
        )
        self.interp_layer = interp_layer

    @staticmethod
    def center_crop(x: th.Tensor, shape: th.Size) -> th.Tensor:
        assert x.shape[:2] == shape[:2]
        shape_dif = tuple(xs - s for xs, s in zip(x.shape, shape))
        slicing = tuple(
            slice(d - d // 2, xs - d // 2) for xs, d in zip(x.shape, shape_dif)
        )
        return x[slicing]

    def forward(self, x: th.Tensor, encoder_features: List[th.Tensor]) -> th.Tensor:
        x = self.first_block(x)
        for feats, block in zip(encoder_features, self.blocks):
            x = self.interp_layer(x, scale_factor=2)
            feats = self.center_crop(feats, x.shape)
            x = th.cat((x, feats), dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        planes: Tuple[int] = (32, 64, 128, 256),
        conv_layer: Callable = nn.Conv2d,
        resize_output: bool = True,
    ):
        super().__init__()

        if conv_layer == nn.Conv2d:
            pool_layer = partial(F.max_pool2d, kernel_size=2)
            self.interp_layer = partial(
                F.interpolate, mode="bilinear", align_corners=False
            )
        elif conv_layer == nn.Conv3d:
            pool_layer = partial(F.max_pool3d, kernel_size=2)
            self.interp_layer = partial(
                F.interpolate, mode="trilinear", align_corners=False
            )
        else:
            raise NotImplementedError

        self._resize_output = resize_output
        self.encoder = Encoder(
            (in_channels,) + planes, conv_layer=conv_layer, pool_layer=pool_layer
        )
        self.decoder = Decoder(
            planes[::-1], conv_layer=conv_layer, interp_layer=self.interp_layer
        )
        self.head = conv_layer(planes[0], out_channels, kernel_size=1)

        self.init_weights()

    def forward(self, x):
        shape = x.shape[2:]
        encoder_feats = self.encoder(x)
        out = self.decoder(encoder_feats[-1], encoder_feats[:-1][::-1])
        out = self.head(out)
        if self._resize_output:
            out = self.interp_layer(out, size=shape)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
