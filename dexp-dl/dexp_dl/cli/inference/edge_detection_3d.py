import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Sequence

import click
import higra as hg
import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from numpy.typing import ArrayLike
from scipy.signal._signaltools import _centered
from toolz import curry
from torch.utils.data import DataLoader
from tqdm import tqdm

from dexp.cli.parsing import multi_devices_option, channels_option, input_dataset_argument
from dexp.datasets.zarr_dataset import ZDataset

from dexp_dl.data import DexpTileDataset
from dexp_dl.models import hrnet, unet
from dexp_dl.models.utils import load_weights
from dexp_dl.postprocessing import hierarchy
from dexp_dl.transforms import gray_normalize


@curry
def transforms(image: ArrayLike, max: Optional[float]) -> th.Tensor:
    if max is None:
        image, _ = gray_normalize(image, None)
    else:
        image = np.clip(image, 0, max)
        image = image / max
    return th.Tensor(image).unsqueeze_(0).half()


def postprocess(image: th.Tensor, interpolate: bool) -> th.Tensor:
    if interpolate:
        image = F.interpolate(
            image, scale_factor=2, mode="trilinear", align_corners=True
        )
    image = th.sigmoid(image)
    return image


def get_prefix(channel: str, n_channels: int) -> str:
    return f"{channel}_" if n_channels > 1 else ""


def write_segmentation(
    dataset: ZDataset,
    time_pt: int,
    prediction: ArrayLike,
    boundary: ArrayLike,
    pred_threshold: float,
    hier_thold: float,
    channel_name: str,
) -> None:
    hier = hierarchy.create_hierarchies(
        prediction > pred_threshold,
        boundary,
        min_area=50,
        cut_threshold=hier_thold,
        min_frontier=0.0,
        hierarchy_fun=hg.watershed_hierarchy_by_volume,
    )
    labels = hierarchy.to_labels(hier, boundary.shape)
    dataset.write_stack(channel_name, time_pt, labels)


def setup(rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup(world_size: int):
    if world_size > 1:
        dist.destroy_process_group()


@curry
def worker_fn(
    rank: int,
    input_path: Path,
    channels: Sequence[str],
    output_path: str,
    network_architecture: str,
    weights_path: Path,
    tile_shape: Tuple[int],
    overlap: Tuple[int],
    hierarchy_threshold: Optional[float],
    prediction_threshold: float,
    max_norm: Optional[float],
    starting_index: int,
    devices: Tuple[int],
    flip_on_odd: bool,
):
    world_size = len(devices)
    device = devices[rank]
    setup(rank, world_size)

    in_ds = ZDataset(str(input_path))
    out_ds = ZDataset(output_path, mode="r+")

    th.cuda.set_device(device)

    if network_architecture == "hrnet":
        net = hrnet.hrnet_w18_small_v2(
            pretrained=False, in_chans=1, num_classes=2, image_ndim=3
        )
        do_interpolation = True
    elif network_architecture == "unet":
        net = unet.UNet(
            in_channels=1, out_channels=2, conv_layer=th.nn.Conv3d, resize_output=True
        )
        do_interpolation = False
    else:
        raise NotImplementedError

    load_weights(net, weights_path)
    net = net.cuda()

    for channel in channels:

        print(f"Predicting channel {channel}")
        in_array = in_ds.get_array(channel)
        prefix = get_prefix(channel, len(channels))

        dataset = DexpTileDataset(
            array=in_array,
            tile_shape=tile_shape,
            overlap=overlap,
            transforms=transforms(max=max_norm),
            starting_index=starting_index,
            rank=rank,
            world_size=world_size,
            flip_odd_index=flip_on_odd,
        )
        loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1)

        print("N time points", dataset.n_time_points, "for rank", rank)
        print("N tiles", dataset.n_tiles)
        keys = ("Prediction", "Boundary")

        net.eval()
        with th.inference_mode():
            with th.cuda.amp.autocast():
                for t, batch in tqdm(loader, desc="Predicting and partitioning"):
                    t = t.item()
                    batch = batch.cuda()

                    pred_batch = net.forward(batch)
                    pred_batch = postprocess(pred_batch, do_interpolation).cpu().numpy()

                    if pred_batch.shape[-3:] != batch.shape[-3:]:
                        # this could happen when the shape has an odd number
                        pred_batch = _centered(
                            pred_batch,
                            tuple(pred_batch.shape[:-3]) + tuple(batch.shape[-3:]),
                        )

                    for preds in pred_batch:
                        results = {}
                        for pred, key in zip(preds, keys):
                            stack = dataset.write_tile(pred, key)
                            if stack is not None:
                                results[key] = stack
                                worker = threading.Thread(
                                    target=out_ds.write_stack, args=(prefix + key, t, stack)
                                )
                                worker.start()

                        if len(results) > 0:
                            if hierarchy_threshold is not None:
                                worker = threading.Thread(
                                    target=write_segmentation,
                                    args=(
                                        out_ds,
                                        t,
                                        results["Prediction"],
                                        results["Boundary"],
                                        prediction_threshold,
                                        hierarchy_threshold,
                                        prefix + "Labels",
                                    ),
                                )
                                worker.start()

    in_ds.close()
    out_ds.close()

    cleanup(world_size)


def _parse_shape(ctx: click.Context, opt: click.Option, value: str) -> Tuple[int]:
    shape = tuple(int(v) for v in value.split(","))
    assert len(shape) == 3
    return shape


def setup_output_dataset(
    input_dataset: ZDataset,
    output_path: str,
    channels: Sequence[str],
    save_segm: bool,
    pred_dtype: np.dtype = np.float16,
) -> None:

    out_ds = ZDataset(output_path, mode="w", parent=input_dataset)

    for channel in channels:
        in_array = input_dataset.get_array(channel)
        prefix = get_prefix(channel, len(channels))

        shape = in_array.shape

        out_ds.add_channel(prefix + "Prediction", shape=shape, dtype=pred_dtype)
        out_ds.add_channel(prefix + "Boundary", shape=shape, dtype=pred_dtype)

        if save_segm:
            out_ds.add_channel(prefix + "Labels", shape=shape, dtype=np.uint32)
    
    out_ds.close()


@click.command()
@input_dataset_argument()
@channels_option()
@click.option("--output-path", "-o", required=True, help="Output dataset path.")
@click.option(
    "--weights-path",
    "-wp",
    required=True,
    type=click.Path(exists=True),
    help="Model weights path.",
)
@click.option(
    "--architecture",
    "-a",
    default="hrnet",
    type=click.Choice(("hrnet", "unet")),
    show_default=True,
)
@click.option(
    "--tile-shape", "-ts", default="144,144,144", type=str, callback=_parse_shape
)
@click.option("--overlap", "-ov", default="32,64,64", type=str, callback=_parse_shape)
@click.option(
    "--hierarchy-threshold",
    "-h",
    default=None,
    type=float,
    help="Watershed hierarchy threshold.",
)
@click.option(
    "--prediction-threshold",
    "-p",
    default=0.5,
    type=float,
    help="Prediction threshold.",
)
@multi_devices_option()
@click.option("--max-norm", "-mn", default=None, type=float)
@click.option("--resume", "-r", is_flag=True, default=False)
@click.option(
    "--flip-on-odd",
    "-flip",
    is_flag=True,
    default=False,
    help="Computes the prediction on the flipped array (and unflips) to avoid aligned tiles.",
)
def cli_inference_edge_3d(
    input_dataset: ZDataset,
    channels: Sequence[str],
    output_path: str,
    weights_path: Path,
    architecture: str,
    tile_shape: Tuple[int],
    overlap: Tuple[int],
    hierarchy_threshold: Optional[float],
    prediction_threshold: float,
    devices: List[int],
    max_norm: Optional[float],
    resume: bool,
    flip_on_odd: bool,
):

    starting_index = 0
    if Path(output_path).exists() and not resume:
        raise ValueError(f"{output_path} exists.")
    elif not Path(output_path).exists() and resume:
        raise ValueError(f"{output_path} not found. Processing cannot be resumed.")
    elif Path(output_path).exists() and resume:
        out_ds = ZDataset(output_path)
        starting_index = min(
            out_ds.first_uninitialized_time_point(ch) for ch in out_ds.channels()
        )
        print(f"Resuming from time point {starting_index}")

    if starting_index == 0:
        setup_output_dataset(input_dataset, output_path, channels, hierarchy_threshold is not None)

    worker = worker_fn(
        input_path=input_dataset.path,
        channels=channels,
        output_path=output_path,
        weights_path=weights_path,
        network_architecture=architecture,
        tile_shape=tile_shape,
        overlap=overlap,
        hierarchy_threshold=hierarchy_threshold,
        prediction_threshold=prediction_threshold,
        max_norm=max_norm,
        starting_index=starting_index,
        devices=devices,
        flip_on_odd=flip_on_odd,
    )

    if len(devices) > 1:
        mp.spawn(worker, nprocs=len(devices))
    else:
        worker(rank=0)


if __name__ == "__main__":
    cli_inference_edge_3d()
