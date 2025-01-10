import warnings
from pathlib import Path

import click
import dask.array as da
import numpy as np
from dask.array.image import imread as da_imread
from dexp.cli.parsing import _parse_chunks
from skimage.segmentation import relabel_sequential
from tifffile import imread, imwrite
from tqdm import tqdm

from dexp_dl.transforms import random_slice


@click.command()
@click.option("--image-path", "-i", required=True, type=click.Path())
@click.option("--label-path", "-l", required=True, type=click.Path())
@click.option("--out-directory", "-o", required=True, type=str)
@click.option("--n-tiles", "-n", required=True, type=int)
@click.option("--tile-shape", "-t", default="(96,96,96)", type=str, show_default=True)
@click.option("--prefix", "-p", default="", type=str)
def cli_tif_to_tiles(
    image_path: str,
    label_path: str,
    out_directory: str,
    n_tiles: int,
    tile_shape: str,
    prefix: str,
) -> None:

    if "*" in image_path:
        image = da_imread(image_path)
        label = da_imread(label_path)
    else:
        image = imread(image_path)
        label = imread(label_path)

    if image.shape != label.shape:
        warnings.warn(
            f"Image shape {image.shape} and {label.shape} doesn't match using smaller one as reference."
        )

    shape = image.shape if np.prod(image.shape) < np.prod(label.shape) else label.shape

    tile_shape = _parse_chunks(tile_shape)

    out_directory = Path(out_directory)
    out_directory.mkdir(exist_ok=True)

    im_dir = out_directory / "images"
    im_dir.mkdir(exist_ok=True)
    lb_dir = out_directory / "labels"
    lb_dir.mkdir(exist_ok=True)

    count = 0
    pbar = tqdm(total=n_tiles, desc="Selecting tiles")
    while count < n_tiles:
        slicing = random_slice(shape, tile_shape)
        lb = label[slicing]

        if isinstance(lb, da.Array):
            lb = lb.compute()

        if (lb != 0).sum() == 0:
            continue

        lb, _, _ = relabel_sequential(lb)
        im = image[slicing]

        if isinstance(im, da.Array):
            im = im.compute()

        imwrite(str(lb_dir / f"{prefix}{count:05d}.tif"), lb)
        imwrite(str(im_dir / f"{prefix}{count:05d}.tif"), im)
        count += 1
        pbar.update()
