import subprocess
from pathlib import Path

import pytest


def test_edge_detection_dexp_end_to_end(nuclei_dexp_directory: Path) -> None:
    subprocess.run(
        [
            "dexp-dl",
            "train",
            "edge-dexp",
            "-i",
            str(nuclei_dexp_directory),
            "-ic",
            "image",
            "-lc",
            "labels",
            "-n",
            "dexptest",
            "-a",
            "unet",
            "-it",
            "200",
            "-e",
            "1",
            "-d",
            "0",
        ]
    ).check_returncode()

    # FIXME: add this as an argument
    weights_path = "logs/dexptest/last.ckpt"

    subprocess.run(
        [
            "dexp-dl",
            "inference",
            "edge",
            "-i",
            str(nuclei_dexp_directory),
            "-c",
            "image",
            "-o",
            str(nuclei_dexp_directory.parent / "pred.zarr"),
            "-a",
            "unet",
            "-wp",
            weights_path,
            "-flip",
            "-d",
            "0",
        ]
    ).check_returncode()


@pytest.mark.parametrize(
    "nuclei_tiles_directory",
    [
        dict(
            z_scale=2,
        )
    ],
    indirect=True,
)
def test_edge_detection_tiles(nuclei_tiles_directory: Path) -> None:
    subprocess.run(
        [
            "dexp-dl",
            "train",
            "edge-tiles",
            "-p",
            str(nuclei_tiles_directory),
            "-n",
            "testing",
            "-e",
            "1",
            "-z",
            "2",
            "-d",
            "0",
        ]
    ).check_returncode()
