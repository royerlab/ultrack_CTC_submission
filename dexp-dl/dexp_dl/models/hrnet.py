# copied and modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py

from typing import Dict

import torch as th
import torch.nn.functional as F
from timm.models.helpers import build_model_with_cfg, default_cfg_for_features

from dexp_dl.models._hrnet import _BN_MOMENTUM
from dexp_dl.models._hrnet import HighResolutionNet as _HighResolutionNet
from dexp_dl.models._hrnet import cfg_cls, default_cfgs


class HighResolutionNet(_HighResolutionNet):
    def __init__(self, cfg: Dict, **kwargs):
        super().__init__(cfg, **kwargs)

        th.backends.cudnn.benchmark = True

        # removing stride to avoid reduction of image size
        self.conv1.stride = 1
        self.conv2.stride = 2  # if self.image_ndim == 2 else 2
        self.final_layer = self._make_final_layer()

        # removing unused modules
        del self.downsamp_modules
        del self.classifier
        del self.global_pool

    def _make_final_layer(self) -> th.nn.Module:
        # last_inp_channels = self.head_channels[3] * Bottleneck.expansion
        conv_layer = self.block_args["conv_layer"]
        norm_layer = self.block_args["norm_layer"]
        return th.nn.Sequential(
            conv_layer(
                in_channels=self.num_features,
                out_channels=self.num_features,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            norm_layer(self.num_features, momentum=_BN_MOMENTUM),
            th.nn.ReLU(inplace=True),
            conv_layer(
                in_channels=self.num_features,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward_features(self, x: th.Tensor) -> th.Tensor:
        inter_feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        inter_feats.append(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        inter_feats.append(x)

        # Stages
        yl = self.stages(x)

        # Classification Head
        yl = [incre_module(y) for incre_module, y in zip(self.incre_modules, yl)]
        if self.image_ndim == 2:
            mode = "bilinear"
        elif self.image_ndim == 3:
            mode = "trilinear"
        else:
            raise NotImplementedError

        yl += inter_feats
        size = yl[0].shape[-self.image_ndim :]
        for i in range(1, len(yl)):
            yl[i] = F.interpolate(yl[i], size=size, mode=mode, align_corners=True)

        y = th.cat(yl, dim=1)
        return y

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.forward_features(x)
        return self.final_layer(y)


def _create_hrnet(variant, pretrained, **model_kwargs):
    if model_kwargs.get("image_ndim", 2) != 2:
        assert pretrained is False
    model_cls = HighResolutionNet
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=cfg_cls[variant],
        **model_kwargs
    )
    model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model


def hrnet_w18_small(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w18_small", pretrained, **kwargs)


def hrnet_w18_small_v2(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w18_small_v2", pretrained, **kwargs)


def hrnet_w18(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w18", pretrained, **kwargs)


def hrnet_w30(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w30", pretrained, **kwargs)


def hrnet_w32(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w32", pretrained, **kwargs)


def hrnet_w40(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w40", pretrained, **kwargs)


def hrnet_w44(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w44", pretrained, **kwargs)


def hrnet_w48(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w48", pretrained, **kwargs)


def hrnet_w64(pretrained=True, **kwargs) -> HighResolutionNet:
    return _create_hrnet("hrnet_w64", pretrained, **kwargs)


def filter_final_layers(parameters: Dict) -> Dict:
    return {k: v for k, v in parameters.items() if "final" not in k}
