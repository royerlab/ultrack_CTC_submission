from pathlib import Path
from typing import Sequence

import click
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom
from dexp.datasets import ZDataset
from tifffile import imread
from tqdm import tqdm


@click.command()
@click.argument("input_paths", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option("--output-path", "-o", type=click.Path(path_type=Path))
@click.option("--lower-quantil", "-lq", type=float, required=True)
@click.option("--upper-quantil", "-uq", type=float, required=True)
@click.option("--outlier", "-out", type=bool, is_flag=True, default=False)
@click.option("--z-scale", "-z", type=int, default=1)
@click.option("--overwrite", "-ow", type=bool, is_flag=True, default=False)
def main(
    input_paths: Sequence[Path],
    output_path: Path,
    lower_quantil: float,
    upper_quantil: float,
    outlier: bool,
    z_scale: int,
    overwrite: bool,
) -> None:

    channel = "Image"
    files = sorted(input_paths)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    ds = ZDataset(output_path, mode="w" if overwrite else "w-")

    # unified memory
    # cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    for t, p in tqdm(enumerate(files)):
        im = cp.asarray(imread(p))
        small_img = im[(slice(None, None, 2),) * im.ndim]
        im = im.astype(np.float32)
        np.subtract(im, np.quantile(small_img, lower_quantil), out=im)
        np.divide(im, np.quantile(small_img, upper_quantil).astype(np.float32), out=im)

        if outlier:
            im[im > 1.5] = 0.0

        np.clip(im, 0, 1, out=im)

        if z_scale > 1:
            im = zoom(im, (z_scale, 1, 1), order=1, mode="nearest")

        if t == 0:
            ds.add_channel(channel, shape=(len(files),) + im.shape, dtype=np.float16)

        ds.write_stack(channel, t, im.get())

    ds.close()


if __name__ == "__main__":
    main()
