import multiprocessing as mp
from typing import List

import click
import cupy
import cupyx.scipy.ndimage as ndi
import zarr
from arbol import aprint
from dexp.cli.parsing import parse_devices
from dexp.datasets import ZDataset
from toolz import curry


@curry
def _process(
    t: int,
    in_array: zarr.Array,
    out_array: zarr.Array,
    z_scale: int,
    devices: List[int],
) -> None:
    device = devices[mp.current_process()._identity[0] - 1]
    with cupy.cuda.Device(device):
        out_array[t] = ndi.zoom(
            cupy.asarray(in_array[t]), zoom=(z_scale, 1, 1), order=1
        ).get()
    aprint(f"Processed time point {t}", end="\r")


@click.command()
@click.option(
    "--input-path",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input dataset path.",
)
@click.option("--output-path", "-o", required=True, help="Output dataset path.")
@click.option("--z-scale", "-z", type=int, help="Ratio between z and xy dimensions.")
@click.option("--devices", "-d", default="0", type=str)
def cli_rescale_z(
    input_path: str,
    output_path: str,
    z_scale: int,
    devices: str,
):
    in_ds = ZDataset(input_path)
    out_ds = ZDataset(output_path, mode="w", parent=in_ds)

    devices = parse_devices(devices)
    pool = mp.Pool(processes=len(devices))

    for ch in in_ds.channels():
        in_array = in_ds.get_array(ch)
        shape = list(in_array.shape)
        shape[1] *= z_scale
        out_array = out_ds.add_channel(ch, shape, dtype=in_array.dtype)

        process = _process(
            in_array=in_array, out_array=out_array, z_scale=z_scale, devices=devices
        )

        pool.map(process, (t for t in range(shape[0])))
        print()

    in_ds.close()
    out_ds.close()
