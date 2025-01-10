from pathlib import Path

import numpy as np
from dexp.datasets.synthetic_datasets import (
    generate_dexp_zarr_dataset,
    generate_nuclei_background_data,
)
from scipy.ndimage import label
from tifffile import imwrite


def nuclei_tiles(
    path: Path, n_tiles: int = 16, size: int = 96, z_scale: int = 1
) -> None:
    rng = np.random.RandomState(42)

    labels_path = path / "labels"
    images_path = path / "images"

    labels_path.mkdir(exist_ok=True)
    images_path.mkdir(exist_ok=True)

    for i in range(n_tiles):
        binary_mask, _, image = generate_nuclei_background_data(
            size, z_scale, rng=rng, dtype=np.uint16
        )
        labels, _ = label(binary_mask)
        imwrite(labels_path / f"{i}.tif", labels)
        imwrite(images_path / f"{i}.tif", image)


def nuclei_dexp(path: Path, n_time_pts: int = 8, **kwargs) -> None:
    rng = np.random.RandomState(42)
    ds = generate_dexp_zarr_dataset(
        path, "nuclei", rng=rng, n_time_pts=n_time_pts, dtype=np.int32, **kwargs
    )
    binary_mask = ds.get_array("ground-truth")
    labels = ds.add_channel("labels", binary_mask.shape, binary_mask.dtype)
    for t in range(binary_mask.shape[0]):
        lb, _ = label(binary_mask[t])
        labels[t] = lb
    ds.close()
