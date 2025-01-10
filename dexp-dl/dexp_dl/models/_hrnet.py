""" HRNet

Copied from https://github.com/rwightman/pytorch-image-models and modified.

Copied from https://github.com/HRNet/HRNet-Image-Classification

Original header:
  Copyright (c) Microsoft
  Licensed under the MIT License.
  Written by Bin Xiao (Bin.Xiao@microsoft.com)
  Modified by Ke Sun (sunk@mail.ustc.edu.cn)
"""
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.features import FeatureInfo
from timm.models.helpers import build_model_with_cfg, default_cfg_for_features
from timm.models.hrnet import cfg_cls, default_cfgs
from timm.models.layers import create_classifier
from timm.models.registry import register_model

from dexp_dl.models._resnet import BasicBlock, Bottleneck
from dexp_dl.models.utils import get_layers

_BN_MOMENTUM = 0.01
_logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        image_ndim: int = 2,
    ):
        super().__init__()

        self.image_ndim = image_ndim
        conv_layer, norm_layer, _ = get_layers(image_ndim)
        self.block_args = dict(conv_layer=conv_layer, norm_layer=norm_layer)

        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        error_msg = ""
        if num_branches != len(num_blocks):
            error_msg = f"NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})"
        elif num_branches != len(num_channels):
            error_msg = (
                f"NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})"
            )
        elif num_branches != len(num_inchannels):
            error_msg = (
                f"NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(num_inchannels)})"
            )
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        conv_layer, norm_layer, _ = get_layers(self.image_ndim)
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                conv_layer(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(
                    num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM
                ),
            )

        layers = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                **self.block_args,
            )
        ]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    **self.block_args,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        conv_layer, norm_layer, _ = get_layers(self.image_ndim)
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            conv_layer(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            norm_layer(num_inchannels[i], momentum=_BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    conv_layer(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    norm_layer(
                                        num_outchannels_conv3x3, momentum=_BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    conv_layer(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    norm_layer(
                                        num_outchannels_conv3x3, momentum=_BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def _maybe_pad(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if x.shape == shape:
            return x
        diff = torch.tensor(shape) - torch.tensor(x.shape)
        left_pad = torch.div(diff, 2, rounding_mode="floor")
        right_pad = diff - left_pad

        if self.image_ndim == 2:
            return F.pad(x, (left_pad[-1], right_pad[-1], left_pad[-2], right_pad[-2]))
        elif self.image_ndim == 3:
            return F.pad(
                x,
                (
                    left_pad[-1],
                    right_pad[-1],
                    left_pad[-2],
                    right_pad[-2],
                    left_pad[-3],
                    right_pad[-3],
                ),
            )
        else:
            raise NotImplementedError

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    outer = self._maybe_pad(fuse_outer[j](x[j]), y.shape)
                    y = y + outer
            x_fuse.append(self.fuse_act(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionNet(nn.Module):
    def __init__(
        self,
        cfg,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        drop_rate=0.0,
        head="classification",
        image_ndim: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.image_ndim = image_ndim

        conv_layer, norm_layer, _ = get_layers(self.image_ndim)
        self.block_args = dict(conv_layer=conv_layer, norm_layer=norm_layer)

        stem_width = cfg["STEM_WIDTH"]
        self.conv1 = conv_layer(
            in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = norm_layer(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = norm_layer(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        self.head = head
        self.head_channels = None  # set if _make_head called
        if head == "classification":
            # Classification Head
            self.num_features = 2048
            (
                self.incre_modules,
                self.downsamp_modules,
                self.final_layer,
            ) = self._make_head(pre_stage_channels)
            self.global_pool, self.classifier = create_classifier(
                self.num_features, self.num_classes, pool_type=global_pool
            )
        elif head == "incre":
            self.num_features = 2048
            self.incre_modules, _, _ = self._make_head(pre_stage_channels, True)
        else:
            self.incre_modules = None
            self.num_features = 256

        curr_stride = 2
        # module names aren't actually valid here, hook or FeatureNet based extraction would not work
        self.feature_info = [dict(num_chs=64, reduction=curr_stride, module="stem")]
        for i, c in enumerate(
            self.head_channels if self.head_channels else num_channels
        ):
            curr_stride *= 2
            c = c * 4 if self.head_channels else c  # head block expansion factor of 4
            self.feature_info += [
                dict(num_chs=c, reduction=curr_stride, module=f"stage{i + 1}")
            ]

        self.init_weights()

    def _make_head(self, pre_stage_channels, incre_only=False):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]
        conv_layer, norm_layer, _ = get_layers(self.image_ndim)

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(
                self._make_layer(
                    head_block, channels, self.head_channels[i], 1, stride=1
                )
            )
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                conv_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                norm_layer(out_channels, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            conv_layer(
                in_channels=self.head_channels[3] * head_block.expansion,
                out_channels=self.num_features,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            norm_layer(self.num_features, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        conv_layer, norm_layer, _ = get_layers(self.image_ndim)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            conv_layer(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            norm_layer(
                                num_channels_cur_layer[i], momentum=_BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            conv_layer(inchannels, outchannels, 3, 2, 1, bias=False),
                            norm_layer(outchannels, momentum=_BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        conv_layer, norm_layer, _ = get_layers(self.image_ndim)
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_layer(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample, **self.block_args)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, **self.block_args))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    image_ndim=self.image_ndim,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def stages(self, x) -> List[torch.Tensor]:
        x = self.layer1(x)

        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.transition2)
        ]
        yl = self.stage3(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.transition3)
        ]
        yl = self.stage4(xl)
        return yl

    def forward_features(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Stages
        yl = self.stages(x)

        # Classification Head
        y = self.incre_modules[0](yl[0])
        for i, down in enumerate(self.downsamp_modules):
            y = self.incre_modules[i + 1](yl[i + 1]) + down(y)
        y = self.final_layer(y)
        return y

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


class HighResolutionNetFeatures(HighResolutionNet):
    """HighResolutionNet feature extraction

    The design of HRNet makes it easy to grab feature maps, this class provides a simple wrapper to do so.
    It would be more complicated to use the FeatureNet helpers.

    The `feature_location=incre` allows grabbing increased channel count features using part of the
    classification head. If `feature_location=''` the default HRNet features are returned. First stem
    conv is used for stride 2 features.
    """

    def __init__(
        self,
        cfg,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        drop_rate=0.0,
        feature_location="incre",
        out_indices=(0, 1, 2, 3, 4),
        image_ndim: int = 2,
    ):
        assert feature_location in ("incre", "")
        super().__init__(
            cfg,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            drop_rate=drop_rate,
            head=feature_location,
            image_ndim=image_ndim,
        )
        self.feature_info = FeatureInfo(self.feature_info, out_indices)
        self._out_idx = {i for i in out_indices}

    def forward_features(self, x):
        assert False, "Not supported"

    def forward(self, x) -> List[torch.tensor]:
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if 0 in self._out_idx:
            out.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.stages(x)
        if self.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.incre_modules)]
        for i, f in enumerate(x):
            if i + 1 in self._out_idx:
                out.append(f)
        return out


def _create_hrnet(variant, pretrained, **model_kwargs):
    model_cls = HighResolutionNet
    features_only = False
    kwargs_filter = None
    if model_kwargs.pop("features_only", False):
        model_cls = HighResolutionNetFeatures
        kwargs_filter = ("num_classes", "global_pool")
        features_only = True
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=cfg_cls[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **model_kwargs,
    )
    if features_only:
        model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model


@register_model
def hrnet_w18_small(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w18_small", pretrained, **kwargs)


@register_model
def hrnet_w18_small_v2(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w18_small_v2", pretrained, **kwargs)


@register_model
def hrnet_w18(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w18", pretrained, **kwargs)


@register_model
def hrnet_w30(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w30", pretrained, **kwargs)


@register_model
def hrnet_w32(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w32", pretrained, **kwargs)


@register_model
def hrnet_w40(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w40", pretrained, **kwargs)


@register_model
def hrnet_w44(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w44", pretrained, **kwargs)


@register_model
def hrnet_w48(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w48", pretrained, **kwargs)


@register_model
def hrnet_w64(pretrained=True, **kwargs):
    return _create_hrnet("hrnet_w64", pretrained, **kwargs)
