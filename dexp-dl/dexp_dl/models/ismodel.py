from typing import Dict, List

import torch as th
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from dexp_dl.models._hrnet import _BN_MOMENTUM
from dexp_dl.models.utils import get_layers


class ISModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_ndim: int = 3,
        in_channels: int = 3,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.backbone = model
        self._image_ndim = image_ndim
        self._in_channels = in_channels
        self._freeze_bn = freeze_bn

        conv_layer, norm_layer, _ = get_layers(image_ndim)

        self.input_block = nn.Sequential(
            conv_layer(in_channels + 3, 9, kernel_size=1),
            norm_layer(9, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv_layer(9, 3, kernel_size=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.input_block(x)
        return self.backbone(x)

    def get_training_parameters(self, base_lr: float, scale: float = 0.1) -> List[Dict]:
        # TODO: delete this when old hrnet code is removed
        # return [
        #     {'params': self.input_block.parameters(), 'lr': base_lr},
        #     {'params': [p for n, p in self.backbone.named_parameters() if 'final_layer' not in n],
        #      'lr': base_lr * scale},
        #     {'params': self.backbone.final_layer.parameters(), 'lr': base_lr},
        # ]
        return [
            {"params": self.input_block.parameters(), "lr": base_lr},
            {
                "params": [
                    p for n, p in self.backbone.named_parameters() if "ocr" not in n
                ],
                "lr": base_lr * scale,
            },
            {
                "params": [
                    p for n, p in self.backbone.named_parameters() if "ocr" in n
                ],
                "lr": base_lr,
            },
        ]

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_bn:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self
