import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import torch as th
import torch.nn.functional as F
from toolz import curry
from torch import nn


def interpolate(input: th.Tensor, **kwargs) -> th.Tensor:
    if "mode" in kwargs and "linear" == kwargs["mode"]:
        if input.ndim == 5:
            kwargs["mode"] = "trilinear"
        elif input.ndim == 4:
            kwargs["mode"] = "bilinear"

    if "align_corners" not in kwargs:
        kwargs["align_corners"] = True

    return F.interpolate(input, **kwargs)


@curry
def dropout_nd(input: th.Tensor, **kwargs) -> th.Tensor:
    if input.ndim == 5:
        return F.dropout3d(input, **kwargs)
    elif len(input) == 4:
        return F.dropout2d(input, **kwargs)
    return F.dropout(input, **kwargs)


@curry
def maxpool_nd(input: th.Tensor, **kwargs) -> th.Tensor:
    if input.ndim == 5:
        return F.max_pool3d(input, **kwargs)
    elif len(input) == 4:
        return F.max_pool2d(input, **kwargs)
    elif len(input) == 3:
        return F.max_pool1d(input, **kwargs)
    else:
        raise NotImplementedError


def get_layers(n_dim: int) -> Tuple[nn.Module, nn.Module, nn.Module]:
    if n_dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d
    elif n_dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d
    else:
        raise NotImplementedError


def _load_state_dict(path: Union[str, Path]) -> Dict[str, th.Tensor]:
    if isinstance(path, str):
        path = Path(path)

    assert path.exists()

    state_dict = th.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    return state_dict


def load_weights(model: nn.Module, path: Union[str, Path]) -> None:
    state_dict = _load_state_dict(path)
    errors = model.load_state_dict(state_dict, strict=False)
    print("\nMissing keys", errors.missing_keys)
    print("\nUnexpected keys", errors.unexpected_keys)


def load_3d_weights_from_2d(model: nn.Module, path: Union[str, Path]) -> None:
    state_dict = _load_state_dict(path)
    load_3d_state_dict_from_2d(model, state_dict)


def load_3d_state_dict_from_2d(
    model: nn.Module, state_dict: Dict[str, th.Tensor]
) -> None:
    with th.no_grad():
        for key, module in model.named_parameters():
            src_module = state_dict.pop(key)
            if src_module.ndim == 4 and module.ndim == 5:
                # silly way to guess if it's conv
                init_3d_conv_from_2d(src_module, module)
            else:
                module.data.copy_(src_module.data)

        for key, module in model.named_buffers():
            module.data.copy_(state_dict.pop(key).data)

    if len(state_dict) != 0:
        warnings.warn(f"Missed the following paramss.\n{state_dict.keys()}")


def init_3d_conv_from_2d(conv2d: th.Tensor, conv3d: th.Tensor) -> None:
    shape = conv2d.shape
    assert shape[:2] == conv3d.shape[:2]
    assert shape[2] == shape[3]
    if shape[2] == 1:
        flat_conv3d = conv3d.data.view(-1)
        flat_conv3d.copy_(conv2d.data.view(-1))
        return

    conv3d.fill_(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mid = shape[2] // 2
            conv3d[i, j, mid, :, :] += conv2d[i, j]
            conv3d[i, j, :, mid, :] += conv2d[i, j]
            conv3d[i, j, :, :, mid] += conv2d[i, j]

    conv3d.div_(3)


def channel_shuffle(x: th.Tensor, groups: int) -> th.Tensor:
    batch_size, num_channels = x.size(0), x.size(1)
    shape = x.shape[2:]
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, *shape)
    x = th.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, *shape)

    return x
