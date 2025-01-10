from pathlib import Path
from typing import Callable, Sequence, Tuple

import click
import torch as th
import torch.nn.functional as F
from dexp.cli.parsing import multi_devices_option
from numpy.typing import ArrayLike
from tifffile import imread
from toolz import curry
from torch.utils.data.dataset import random_split

from dexp_dl.cli.utils import (
    edge_loss_option,
    model_option,
    output_transform_option,
    weights_path_option,
)
from dexp_dl.cli.training.utils import edge_dilation_option
from dexp_dl.data import FileDataset
from dexp_dl.training import generic_multivariate_3d
from dexp_dl.transforms import (
    flip_axis,
    gray_normalize,
    random_crop,
    random_noise,
    random_power,
    upsample,
)


@curry
def test_transforms(
    im: ArrayLike, lb: ArrayLike, z_scale: int, output_transform: Callable
) -> Tuple[th.Tensor, th.Tensor]:

    im = im.squeeze()
    lb = lb.squeeze()
    if z_scale != 1:
        im, lb = upsample(im, lb, scale=(z_scale, 1, 1))
    im, lb = random_crop(im, lb, (96, 96, 96))
    im, lb = gray_normalize(im, lb)
    im, lb = output_transform(im, lb)
    return th.Tensor(im).unsqueeze(0), th.Tensor(lb)


@curry
def train_transforms(
    im: ArrayLike, lb: ArrayLike, z_scale: int, output_transform: Callable
) -> Tuple[th.Tensor, th.Tensor]:

    assert im.ndim == 3 and lb.ndim == 3
    im = im.squeeze()
    lb = lb.squeeze()
    im, lb = flip_axis(im, lb, (1, 2))
    if z_scale != 1:
        im, lb = upsample(im, lb, scale=(z_scale, 1, 1))
    im, lb = random_crop(im, lb, (96, 96, 96))
    im, lb = random_power(im, lb)
    im, lb = gray_normalize(im, lb)
    im, lb = random_noise(im, lb)
    im, lb = output_transform(im, lb)
    return th.Tensor(im).unsqueeze(0), th.Tensor(lb)


@click.command()
@click.option(
    "--tiles-directory-path", "-p", type=click.Path(exists=True), required=True
)
@click.option("--images-subdir", "-i", type=str, default="images", show_default=True)
@click.option("--labels-subdir", "-l", type=str, default="labels", show_default=True)
@click.option("--file-extension", "-f", type=str, default="*.tif", show_default=True)
@click.option("--exp-name", "-n", type=str, required=True)
@click.option("--epochs", "-e", type=int, default=25, show_default=True)
@click.option("--z-scale", "-z", type=int, default=1, show_default=True)
@click.option("--test-split-perc", "-t", type=float, default=0.1, show_default=True)
@multi_devices_option()
@model_option()
@weights_path_option()
@edge_loss_option()
@output_transform_option()
@edge_dilation_option()
def cli_train_edge_3d_tiles(
    tiles_directory_path: str,
    images_subdir: str,
    labels_subdir: str,
    file_extension: str,
    exp_name: str,
    epochs: int,
    devices: Sequence[int],
    z_scale: int,
    test_split_perc: float,
    model: th.nn.Module,
    output_transform: Callable,
    edge_loss: Callable,
) -> None:

    tiles_directory_path = Path(tiles_directory_path)

    dataset = FileDataset(
        images_dir=tiles_directory_path / images_subdir,
        labels_dir=tiles_directory_path / labels_subdir,
        file_key=file_extension,
        loader=imread,
    )

    test_size = int(round(len(dataset) * test_split_perc))
    train_ds, test_ds = random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=th.Generator().manual_seed(42),
    )

    train_ds.dataset.transforms = train_transforms(
        z_scale=z_scale, output_transform=output_transform
    )
    test_ds.dataset.transforms = test_transforms(
        z_scale=z_scale, output_transform=output_transform
    )

    losses = [
        edge_loss,
        F.binary_cross_entropy_with_logits,
    ]

    generic_multivariate_3d(
        train_ds=train_ds,
        val_ds=test_ds,
        model=model,
        num_epochs=epochs,
        losses=losses,
        exp_name=exp_name,
        devices=devices,
    )


if __name__ == "__main__":
    cli_train_edge_3d_tiles()
