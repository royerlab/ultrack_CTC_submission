import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import click
import cucim.skimage.morphology as morph
import cupy as cp
import cupyx.scipy.ndimage as ndi
import higra as hg
import numpy as np
from cucim.skimage import filters
from dexp.datasets import ZDataset
from tqdm import tqdm
from ultrack.cli.utils import tuple_callback


def foreground_and_edge_detection(
    array: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    perc_thold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detects foreground and edges regions"""
    array = cp.asarray(array)
    DoG = ndi.gaussian_filter(
        array, sigma=sigma_1, mode="nearest"
    ) - ndi.gaussian_filter(array, sigma=sigma_2, mode="nearest")

    mask = DoG > filters.threshold_otsu(DoG) * perc_thold
    del DoG

    morph.remove_small_objects(mask, min_size=128, out=mask)
    mask = morph.binary_dilation(mask, morph.ball(1)).get()

    # inverting and normalizing
    array = ndi.gaussian_filter(array, sigma=sigma_1, mode="nearest")
    edges = array.max() - array

    del array
    edges = edges - edges.min()
    edges = edges / edges.max()
    edges = edges.get()

    return mask, edges


def segment(
    array: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    perc_thold: float,
    cut_threshold: float,
    min_area: float,
    max_area: float,
    watershed: Callable,
    padding: Optional[int],
    debug: bool,
) -> np.ndarray:

    # array = cp.asarray(array, dtype=np.float16)
    # detection = foreground_detection(array, sigma_1, sigma_2).get()
    # edges = edge_detection(array, edge_sigma).get()
    # array = array.get()
    detection, edges = foreground_and_edge_detection(
        array, sigma_1, sigma_2, perc_thold
    )

    if padding is not None:
        detection = detection[padding:-padding]
        edges = edges[padding:-padding]

    # hiers = create_hierarchies(
    #     detection,
    #     edges,
    #     cache=True,
    #     cut_threshold=cut_threshold,
    #     hierarchy_fun=watershed,
    #     min_frontier=0.0,
    #     max_area=max_area,
    #     min_area=min_area,
    # )

    # labels = to_labels(hiers, edges.shape)
    labels = np.zeros_like(detection, dtype=np.int32)  # FIXME

    if padding is not None:
        _labels = np.zeros(array.shape, dtype=np.int32)
        _labels[padding:-padding] = labels
        labels = _labels

        _edges = np.ones(array.shape, dtype=np.float16)
        _edges[padding:-padding] = edges
        edges = _edges

        _detection = np.zeros(array.shape, dtype=bool)
        _detection[padding:-padding] = detection
        detection = _detection

    if debug:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(
            edges,
            name="EDGES",
            blending="additive",
            colormap="magma",
        )
        viewer.add_labels(detection, name="DETECTION")
        viewer.add_image(array, name="IMAGE", blending="additive")
        viewer.add_labels(labels, name="LABELS")
        napari.run()

    return labels.astype(np.int32), detection, edges


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sigma-1", "-s1", type=str, callback=tuple_callback(dtype=float, length=3)
)
@click.option(
    "--sigma-2", "-s2", type=str, callback=tuple_callback(dtype=float, length=3)
)
@click.option("--perc-thold", "-pt", type=float, show_default=True, default=1.0)
@click.option("--cut", "-c", type=float)
@click.option("--min-area", "-m", type=int)
@click.option("--max-area", "-M", type=int)
@click.option("--overwrite", "-ow", type=bool, is_flag=True, default=False)
@click.option("--padding", "-p", type=int, default=None)
@click.option("--debug", type=bool, is_flag=True, default=False)
@click.option(
    "--watershed", "-ws", type=click.Choice(["area", "volume"]), default="area"
)
def main(
    input_path: Path,
    sigma_1: Union[Sequence[float], float],
    sigma_2: Union[Sequence[float], float],
    cut: float,
    min_area: int,
    max_area: int,
    perc_thold: float,
    watershed: str,
    padding: Optional[int],
    overwrite: bool,
    debug: bool,
) -> None:

    aux_channels = ("Labels", "Prediction", "Boundary")
    if not debug:
        for ch in aux_channels:
            aux_path = input_path / ch
            if aux_path.exists():
                if overwrite:
                    shutil.rmtree(aux_path)
                else:
                    raise ValueError(f"{aux_path} already exists")

    watershed = {
        "area": hg.watershed_hierarchy_by_area,
        "volume": hg.watershed_hierarchy_by_volume,
        "dynamics": hg.watershed_hierarchy_by_dynamics,
    }[watershed]

    ds = ZDataset(input_path, mode="r+")
    array = ds.get_array("Image")
    if not debug:
        labels = ds.add_channel(aux_channels[0], shape=array.shape, dtype=np.int32)
        detections = ds.add_channel(
            aux_channels[1], shape=array.shape, dtype=np.float16
        )
        edges = ds.add_channel(aux_channels[2], shape=array.shape, dtype=np.float16)

    with cp.cuda.Device(0):
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

        for t in tqdm(range(array.shape[0])):
            lb, det, edge = segment(
                array[t],
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                cut_threshold=cut,
                min_area=min_area,
                max_area=max_area,
                perc_thold=perc_thold,
                padding=padding,
                watershed=watershed,
                debug=debug,
            )
            if not debug:
                labels[t] = lb
                detections[t] = det
                edges[t] = edge

    ds.close()


if __name__ == "__main__":
    main()
