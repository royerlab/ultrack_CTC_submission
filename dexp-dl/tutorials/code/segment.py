from pathlib import Path

import click
import cupy as cp
import numpy as np
from cucim.skimage import morphology as morph
from cucim.skimage.filters import threshold_otsu
from dexp.datasets import ZDataset
from dexp.processing.morphology import area_white_top_hat
from pyift import shortestpath as sp
from skimage import segmentation
from tifffile import imwrite
from tqdm import tqdm


@click.command()
@click.option(
    "--image-path", "-i", type=str, required=True, help="Input DEXP dataset path."
)
@click.option(
    "--out-path", "-o", type=str, required=True, help="Directory of output .tif images."
)
@click.option(
    "--channel", "-c", type=str, required=True, help="Image data channel name."
)
@click.option(
    "--step", "-s", type=int, required=True, help="Iteration step over time axis."
)
@click.option(
    "--z-scale",
    "-z",
    type=float,
    default=1.0,
    show_default=True,
    help="Relative scale of z-axis in comparison to x and y.",
)
@click.option(
    "--area-threshold",
    "-a",
    type=float,
    default=1e4,
    show_default=True,
    help="Relative volume threshold for cell detection.",
)
@click.option(
    "--min-area",
    "-m",
    type=int,
    default=200,
    show_default=True,
    help="Minimum area of segments",
)
@click.option(
    "--h-minima",
    "-h",
    type=float,
    default=0.08,
    help="Parameter for cell splitting, greater values produce less and larger cells.",
)
@click.option(
    "--compactness",
    "-cp",
    type=float,
    default=0.003,
    help="Parameter that penalizes cell shape, greater values produce smaller and more spherical cells.",
)
@click.option("--display", "-d", is_flag=True, help="Display results before finishing.")
def main(
    image_path: str,
    out_path: str,
    channel: str,
    step: int,
    z_scale: float,
    area_threshold: float,
    min_area: int,
    h_minima: float,
    compactness: float,
    display: bool,
) -> np.ndarray:

    array = ZDataset(image_path).get_array(channel)
    n = array.shape[0]

    outdir = Path(out_path)
    outdir.mkdir(exist_ok=True)

    imdir = outdir / "images"
    imdir.mkdir(exist_ok=True)
    lbdir = outdir / "labels"
    lbdir.mkdir(exist_ok=True)

    for t in tqdm(range(0, n, step)):
        im = array[t]
        lb = segment(
            im,
            z_scale=z_scale,
            area_threshold=area_threshold,
            min_area=min_area,
            h_minima=h_minima,
            compactness=compactness,
            display=display,
        )
        imwrite(str(imdir / f"{t:03d}.tif"), im)
        imwrite(str(lbdir / f"{t:03d}.tif"), lb)


def segment(
    image: np.ndarray,
    *,
    z_scale: float,
    area_threshold: float,
    min_area: int,
    h_minima: float,
    compactness: float,
    display: bool,
) -> np.ndarray:
    assert image.ndim == 3, f"Image must be 3D, {image.ndim} found."

    image = cp.asarray(image)
    filtered = morph.closing(image, morph.ball(np.sqrt(2)))
    wth = cp.asarray(
        area_white_top_hat(filtered.get(), area_threshold, sampling=1, axis=0)
    )
    image = image.get()

    detection = filtered > threshold_otsu(filtered)
    detection = morph.closing(detection, morph.ball(np.sqrt(2)))
    detection = morph.remove_small_objects(detection, min_size=min_area).get()

    basins = filtered / np.quantile(filtered, 0.999)
    basins = basins.max() - basins
    basins = np.sqrt(basins)
    _, labels = sp.watershed_from_minima(
        basins.get(),
        detection,
        H_minima=h_minima,
        compactness=compactness,
        scales=(z_scale, 1, 1),
    )
    labels[labels < 0] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)

    if display:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(image, name="Input image")
        viewer.add_image(filtered.get(), name="Morph filtered")
        viewer.add_image(wth.get(), name="WTH")
        viewer.add_labels(detection, name="Detection")
        viewer.add_labels(labels, name="Segments")

        napari.run()

    return labels.astype(np.uint16)


if __name__ == "__main__":
    main()
