from pathlib import Path
from typing import Callable

import click
import torch as th
from toolz import curry

from dexp_dl import loss
from dexp_dl.models import hrnet, unet
from dexp_dl.models.utils import load_weights
from dexp_dl.transforms.transforms import add_boundary, add_edt


def model_callback(ctx: click.Context, opt: click.Option, value: str) -> th.nn.Module:
    if value == "hrnet":
        model = hrnet.hrnet_w18_small_v2(
            pretrained=False, in_chans=1, num_classes=2, image_ndim=3
        )
    elif value == "unet":
        model = unet.UNet(
            in_channels=1, out_channels=2, conv_layer=th.nn.Conv3d, resize_output=False
        )
    else:
        raise NotImplementedError(value)

    return model


def model_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--architecture",
            "-a",
            "model",
            default="hrnet",
            type=click.Choice(("hrnet", "unet")),
            show_default=True,
            help="Neural network architecture",
            is_eager=True,
            callback=model_callback,
        )(f)

    return decorator


def weights_path_callback(
    ctx: click.Context, opt: click.Option, value: str
) -> th.nn.Module:
    if value is not None:
        load_weights(ctx.params["model"], value)


def weights_path_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--weights-path",
            "-wp",
            default=None,
            type=click.Path(exists=True, path_type=Path),
            help="Neural network weights path, must match architecture",
            expose_value=False,
            callback=weights_path_callback,
        )(f)

    return decorator


def edge_loss_callback(ctx: click.Context, opt: click.Option, value: str) -> Callable:

    if value == "bce":
        edge_loss_func = curry(loss.weighted_bce_with_logits)(obj_weight=0.2)
    elif value == "dice":

        def edge_loss_func(pred, target):
            return loss.dice_with_logits(pred, target) * 1

    else:
        raise NotImplementedError

    return edge_loss_func


def edge_loss_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--edge-loss",
            "-el",
            default="bce",
            type=click.Choice(("bce", "dice")),
            show_default=True,
            help="Edge prediction loss function",
            callback=edge_loss_callback,
        )(f)

    return decorator


def output_transform_callback(
    ctx: click.Context, opt: click.Option, value: str
) -> Callable:

    if value == "edt":
        output_transform = add_edt
    elif value == "boundary":
        output_transform = add_boundary
    else:
        raise NotImplementedError(value)

    return output_transform


def output_transform_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-mode",
            "-om",
            "output_transform",
            default="edt",
            type=click.Choice(("edt", "boundary")),
            show_default=True,
            help="Output segmentation labels transformation",
            callback=output_transform_callback,
        )(f)

    return decorator
