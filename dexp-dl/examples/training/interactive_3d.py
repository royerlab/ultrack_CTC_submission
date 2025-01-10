from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch as th
from numpy.typing import ArrayLike
from tifffile import imread
from torch import nn

from dexp_dl.data import ISTileDataset
from dexp_dl.models import hrnet_ocr, utils
from dexp_dl.models.ismodel import ISModel
from dexp_dl.training.istrainer import train_interactive_segmentation
from dexp_dl.transforms import (
    flip_axis,
    gray_to_rgb,
    random_crop,
    random_power,
    random_transpose,
)


def val_transforms(im: ArrayLike, lb: ArrayLike) -> Tuple[th.Tensor, th.Tensor]:
    lb = (lb != 0).astype(np.int32)
    im, lb = random_crop(im, lb, (64, 64, 64))
    im -= im.min()
    im = im / float(im.max())
    im, lb = gray_to_rgb(im, lb)
    return th.Tensor(im.astype(np.float16)), th.Tensor(lb).unsqueeze_(0)


def train_transforms(im: ArrayLike, lb: ArrayLike) -> Tuple[th.Tensor, th.Tensor]:
    im, lb = random_transpose(im, lb)
    im, lb = flip_axis(im, lb, (0, 1, 2))
    im, lb = random_power(im, lb)
    return val_transforms(im, lb)


@click.command()
@click.option("--train-directory", "-t", type=click.Path(exists=True), required=True)
@click.option("--val-directory", "-v", type=click.Path(exists=True), required=True)
@click.option("--weights", "-w", type=click.Path(exists=True), required=True)
@click.option("--epochs", "-e", type=int, default=6)
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--num-workers", "-nw", type=int, default=4)
@click.option("--max-num-clicks", "-mnc", type=int, default=7)
def main(
    train_directory: str,
    val_directory: str,
    weights: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    max_num_clicks: int,
):
    train_directory = Path(train_directory)
    val_directory = Path(val_directory)

    train_ds = ISTileDataset(
        images_dir=train_directory / "images",
        labels_dir=train_directory / "labels",
        file_key="*.tif",
        loader=imread,
        transforms=train_transforms,
        max_num_clicks=max_num_clicks,
    )
    val_ds = ISTileDataset(
        images_dir=val_directory / "images",
        labels_dir=val_directory / "labels",
        file_key="*.tif",
        loader=imread,
        transforms=val_transforms,
        max_num_clicks=max_num_clicks,
    )

    backbone = hrnet_ocr.hrnet_18_small(
        second_stride=1,
        conv_layer=nn.Conv3d,
        norm_layer=partial(nn.BatchNorm3d, momentum=0.01),
    )
    model = ISModel(backbone, image_ndim=3)
    utils.load_3d_weights_from_2d(model, path=weights)

    train_interactive_segmentation(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        batch_size=batch_size,
        input_size=128,
        click_radius=1,
        precision=16,
        is_rgb=False,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
