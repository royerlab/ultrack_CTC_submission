from typing import Any, Callable, Dict, Sequence, Tuple

import click
import torch as th
import torch.nn.functional as F
from dexp.cli.parsing import multi_devices_option
from dexp.datasets.zarr_dataset import ZDataset
from numpy.typing import ArrayLike
from toolz import curry

from dexp_dl.cli.utils import (
    edge_loss_option,
    model_option,
    output_transform_option,
    weights_path_option,
)
from dexp_dl.data import DexpQueueDataset
from dexp_dl.data.utils import ConcatIterableDataset
from dexp_dl.training import generic_multivariate_3d
from dexp_dl.transforms import flip_axis, random_power
from dexp_dl.cli.training.utils import edge_dilation_option
from dexp_dl.loss import dice_with_logits



@curry
def split_data_dict(
    data: Dict[str, ArrayLike], input_channels: Sequence[int], labels_channel: str
) -> Tuple[ArrayLike, ArrayLike]:
    return data[input_channels], data[labels_channel]


@curry
def test_transforms(
    data: Dict[str, ArrayLike], split_fun: Callable, output_transform: Callable
) -> Tuple[th.Tensor, th.Tensor]:
    im, lb = split_fun(data)
    im, lb = output_transform(im, lb)
    return th.Tensor(im).unsqueeze(0).half(), th.Tensor(lb).half()


@curry
def train_transforms(
    data: Dict[str, ArrayLike], split_fun: Callable, output_transform: Callable
) -> Tuple[th.Tensor, th.Tensor]:
    im, lb = split_fun(data)
    assert im.ndim == 3 and lb.ndim == 3
    im, lb = flip_axis(im, lb, (0, 1, 2))
    im, lb = random_power(im, lb)
    im, lb = output_transform(im, lb)
    return th.Tensor(im).unsqueeze(0).half(), th.Tensor(lb).half()


@curry
def validation_fun(data: Dict[str, Any], channel: str):
    # check if there is a segments in the mask
    return (data[channel][0] != 0).sum() > 500


@click.command()
@click.option("--input-datasets", "-i", type=str, required=True)
@click.option("--input-channels", "-ic", type=str, required=True)
@click.option("--labels-channel", "-lc", type=str, required=True)
@click.option("--exp-name", "-n", type=str, required=True)
@click.option("--iterations", "-it", type=int, default=1000, show_default=True)
@click.option("--epochs", "-e", type=int, default=25, show_default=True)
@multi_devices_option()
@model_option()
@weights_path_option()
@edge_loss_option()
@output_transform_option()
@edge_dilation_option()
def cli_train_edge_3d_dexp(
    input_datasets: str,
    input_channels: str,
    labels_channel: str,
    exp_name: str,
    iterations: int,
    epochs: int,
    devices: Sequence[int],
    model: th.nn.Module,
    output_transform: Callable,
    edge_loss: Callable,
) -> None:

    # NOTE: train and test split are not done correctly, we don't assert that tiles area independent
    zdatasets = [ZDataset(path) for path in input_datasets.split(",")]

    losses = [
        edge_loss,
        dice_with_logits,
    ]

    split_fun = split_data_dict(
        input_channels=input_channels, labels_channel=labels_channel
    )

    train_tr = train_transforms(split_fun=split_fun, output_transform=output_transform)
    test_tr = test_transforms(split_fun=split_fun, output_transform=output_transform)
    val_fun = validation_fun(channel=labels_channel)
    patch_size = (96, 96, 96)

    train_ds = []
    test_ds = []
    for zds in zdatasets:
        train_ds.append(
            DexpQueueDataset(
                zds,
                [input_channels, labels_channel],
                iterations // len(devices),
                validation_fun=val_fun,
                patch_size=patch_size,
                transforms=train_tr,
                queue_iterations=48,
            )
        )

        test_ds.append(
            DexpQueueDataset(
                zds,
                [input_channels, labels_channel],
                iterations // 10 // len(devices),
                validation_fun=val_fun,
                patch_size=patch_size,
                transforms=test_tr,
            )
        )

    train_ds = ConcatIterableDataset(train_ds)
    test_ds = ConcatIterableDataset(test_ds)

    generic_multivariate_3d(
        train_ds,
        test_ds,
        model=model,
        num_epochs=epochs,
        losses=losses,
        exp_name=exp_name,
        devices=devices,
    )


if __name__ == "__main__":
    cli_train_edge_3d_dexp()
