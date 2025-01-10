import shutil
from pathlib import Path

import click
import numpy as np
from dexp.datasets import ZDataset
from ultrack.imgproc.flow import (
    advenct_from_quasi_random,
    timelapse_flow,
    trajectories_to_tracks,
)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--iterations", "-i", type=int, default=2_000)
@click.option("--output-path", "-o", type=click.Path(path_type=Path), required=True)
@click.option("--overwrite", "-ow", is_flag=True, default=False)
def main(
    input_path: Path,
    output_path: Path,
    iterations: int,
    overwrite: bool,
) -> None:

    ds = ZDataset(input_path, mode="r+")
    array = ds.get_array("Image")

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise ValueError(f"{output_path} already exists. Add `-ow` to overwrite.")

    n_scales = 2 if array.shape[1] <= 16 else 3
    output = timelapse_flow(
        array,
        output_path,
        lr=0.001,
        n_scales=n_scales,
        num_iterations=iterations,
    )

    trajectories = advenct_from_quasi_random(output, array.shape[-3:], n_samples=5_000)

    ds.close()

    tracks = trajectories_to_tracks(trajectories)
    name = output_path.name.removesuffix(".zarr")
    np.save(output_path.parent / f"{name}.npy", tracks)

    print("DONE")


if __name__ == "__main__":
    main()
