# From: https://github.com/saic-vul/ritm_interactive_segmentation
# Modified by Jordao Bragantini
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F

from dexp_dl.models.utils import dropout_nd, interpolate, maxpool_nd


class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c = probs.size(0), probs.size(1)
        n_spatial_dim = int(probs.ndim - 2)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = (
            torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(-1)
        )  # batch x k x c
        if n_spatial_dim == 3:
            ocr_context.unsqueeze_(-1)
        return ocr_context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        out_channels,
        scale=1,
        dropout=0.1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        align_corners=True,
    ):
        super().__init__()
        self.object_context_block = ObjectAttentionBlockND(
            in_channels, key_channels, scale, conv_layer, norm_layer, align_corners
        )
        _in_channels = 2 * in_channels

        self.conv_bn = nn.Sequential(
            conv_layer(
                _in_channels, out_channels, kernel_size=1, padding=0, bias=False
            ),
            nn.Sequential(norm_layer(out_channels), nn.ReLU(inplace=True)),
        )
        self.dropout = dropout_nd(p=dropout)

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.dropout(self.conv_bn(torch.cat([context, feats], 1)))

        return output


class ObjectAttentionBlockND(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        scale=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        align_corners=True,
    ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners

        self.pool = maxpool_nd(kernel_size=scale)
        self.f_pixel = nn.Sequential(
            conv_layer(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
            conv_layer(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
        )
        self.f_object = nn.Sequential(
            conv_layer(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
            conv_layer(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
        )
        self.f_down = nn.Sequential(
            conv_layer(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
        )
        self.f_up = nn.Sequential(
            conv_layer(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sequential(norm_layer(self.in_channels), nn.ReLU(inplace=True)),
        )

    def forward(self, x, proxy):
        # batch_size, h, w = x.size(0), x.size(2), x.size(3)
        batch_size = x.size(0)
        shape = x.shape[2:]
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *shape)
        context = self.f_up(context)
        if self.scale > 1:
            context = interpolate(
                input=context,
                size=shape,
                mode="linear",
                align_corners=self.align_corners,
            )

        return context
