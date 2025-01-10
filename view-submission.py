import warnings
from pathlib import Path

import click
import napari
import numpy as np
from dask.array.image import imread as da_imread
from skimage.measure import regionprops
from tifffile import imread
from tqdm import tqdm


def load_tracks(tracks_dir: str, viewer: napari.Viewer, **kwargs) -> None:
    tracks_dir = Path(tracks_dir)
    tracks_paths = sorted(list(tracks_dir.glob("*.tif")))

    tracks = []
    for t, p in tqdm(enumerate(tracks_paths)):
        mask = imread(p)
        for prop in regionprops(mask):
            tracks.append([prop.label, t, *prop.centroid])

    tracks = np.asarray(tracks)
    track_ids = np.unique(tracks[:, 0]).astype(int).tolist()
    graph = {i: [] for i in track_ids}
    graph_path = list(tracks_dir.glob("*_track.txt"))[0]
    with open(graph_path) as f:
        for line in f.readlines():
            l, b, e, p = line.split(" ")
            l, p = int(l), int(p)
            if p != 0:
                graph[l] = [p]

    try:
        viewer.add_tracks(tracks, graph=graph, **kwargs)
    except ValueError as e:  # noqa
        warnings.warn(e)


@click.command()
@click.argument("dir_path", type=click.Path(exists=True, path_type=Path))
@click.argument("dataset_num", type=str)
@click.option("--z-scale", "-z", default=1.0, type=float)
def main(dir_path: Path, dataset_num: str, z_scale: float) -> None:
    scale = (z_scale, 1, 1)
    viewer = napari.Viewer()

    try:
        im = da_imread(f"{dir_path}/{dataset_num}/*.tif")
        viewer.add_image(im, scale=scale, colormap="magma")
    except ValueError as e:
        print(e)

    tracks_segm = da_imread(f"{dir_path}/{dataset_num}_RES/*.tif")
    viewer.add_labels(tracks_segm, scale=scale)

    load_tracks(f"{dir_path}/{dataset_num}_RES", viewer, name="res tracks", scale=scale)

    segm_dir = Path(dir_path) / "CSB"
    if segm_dir.exists():
        segm = da_imread(f"{segm_dir}/{dataset_num}_RES/*.tif")
        viewer.add_labels(segm, scale=scale)

    napari.run()


if __name__ == "__main__":
    main()
