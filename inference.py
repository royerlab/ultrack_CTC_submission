import threading
import time
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch as th
import torch.nn.functional as F
from cupyx.scipy.ndimage import gaussian_filter
from dexp.datasets import ZDataset
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import CupyBackend
from dexp_dl.data import DexpTileDataset
from dexp_dl.models.utils import load_weights
from scipy.signal._signaltools import _centered
from toolz import curry
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultrack.cli.utils import tuple_callback

from unet import EdgeUNet


def blur_and_write(
    ds: ZDataset,
    channel: str,
    sigma: Tuple[float],
    time: int,
    stack: np.ndarray,
    noise_max: float,
) -> None:
    with CupyBackend() as bkd:
        func = curry(gaussian_filter, sigma=sigma)

        if stack.size > (2**15):
            out = scatter_gather_i2i(stack, func, tiles=320, margins=32)
        else:
            out = bkd.to_numpy(func(bkd.to_backend(stack)))

        if noise_max > 0:
            out += np.random.uniform(0, noise_max, size=out.shape)

        ds.write_stack(channel, time, out)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-path", "-o", type=click.Path(path_type=Path))
@click.option("--channel", "-ch", type=str, default="Image")
@click.option("--weights-path", "-wp", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--overlap-shape", "-os", type=str, callback=tuple_callback(dtype=int, length=3)
)
@click.option(
    "--tile-shape", "-ts", type=str, callback=tuple_callback(dtype=int, length=3)
)
@click.option("--sigma", "-s", type=str, callback=tuple_callback(dtype=float, length=3))
@click.option("--rank", "-r", type=int, default=0, show_default=True)
@click.option("--world-size", "-ws", type=int, default=1, show_default=True)
@click.option("--overwrite", "-ow", type=bool, is_flag=True, default=False)
@click.option("--noise-max", "-nm", type=float, default=0)
def main(
    input_path: Path,
    output_path: Path,
    channel: str,
    weights_path: Path,
    tile_shape: Tuple[int],
    overlap_shape: Tuple[int],
    sigma: Tuple[float],
    rank: int,
    world_size: int,
    overwrite: bool,
    noise_max: float,
) -> None:

    in_ds = ZDataset(input_path)

    if rank == 0:
        out_ds = ZDataset(output_path, mode="w" if overwrite else "w-", parent=in_ds)

        in_array = in_ds.get_array(channel)
        shape = in_array.shape
        out_ds.add_channel("Prediction", shape=shape, dtype=np.float16)
        out_ds.add_channel("Boundary", shape=shape, dtype=np.float16)

    else:
        time.sleep(5)
        out_ds = ZDataset(output_path, mode="r+")

    net = EdgeUNet(
        in_channels=1, out_channels=2, conv_layer=th.nn.Conv3d, kernel_size=5
    )
    load_weights(net, weights_path)
    net = net.cuda()

    in_array = in_ds.get_array(channel)

    dataset = DexpTileDataset(
        array=in_array,
        tile_shape=tile_shape,
        overlap=overlap_shape,
        transforms=None,
        starting_index=0,
        rank=rank,
        world_size=world_size,
        flip_odd_index=False,
    )
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    print("N time points", dataset.n_time_points, "for rank", rank)
    print("N tiles", dataset.n_tiles)
    keys = ("Prediction", "Boundary")

    np.random.seed(rank)

    min_size = 48
    net.eval()
    with th.inference_mode():
        with th.cuda.amp.autocast():
            for t, batch in tqdm(loader, desc="Predicting and partitioning"):
                t = t.item()
                batch = batch[None, ...].cuda()

                pad = None
                if any(s < min_size for s in batch.shape[2:]):
                    diff = np.maximum(min_size - np.asarray(batch.shape[2:]), 0)
                    pad = np.empty(len(diff) * 2, dtype=int)
                    pad[::2] = diff // 2
                    pad[1::2] = diff - pad[::2]
                    batch = F.pad(batch, tuple(reversed(pad)))

                pred_batch = th.sigmoid(net.forward(batch)[0]).cpu().numpy()

                if pad is not None:
                    slicing = tuple((slice(None), slice(None))) + tuple(
                        slice(b, -e) for b, e in zip(pad[::2], pad[1::2]) if e > 0
                    )
                    pred_batch = pred_batch[slicing]

                if pred_batch.shape[-3:] != batch.shape[-3:]:
                    # this could happen when the shape has an odd number
                    pred_batch = _centered(
                        pred_batch,
                        tuple(pred_batch.shape[:-3]) + tuple(batch.shape[-3:]),
                    )

                # pred_batch[:, 1] += 1.0 - pred_batch[:, 0]
                for preds in pred_batch:
                    for pred, key in zip(preds, keys):
                        stack = dataset.write_tile(pred, key)
                        if stack is not None:
                            if key == "Prediction":
                                worker = threading.Thread(
                                    target=out_ds.write_stack, args=(key, t, stack)
                                )
                            else:
                                worker = threading.Thread(
                                    target=blur_and_write,
                                    args=(out_ds, key, sigma, t, stack, noise_max),
                                )
                            worker.start()

    in_ds.close()
    out_ds.close()


if __name__ == "__main__":
    main()
