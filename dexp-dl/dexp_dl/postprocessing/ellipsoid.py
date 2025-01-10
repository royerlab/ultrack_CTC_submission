import warnings
from typing import Tuple

import click
import numpy as np
from dexp.datasets import ZDataset
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from tqdm import tqdm

try:
    import cvxpy as cp
except ImportError:
    pass


def GetHull(points: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ConvexHull]:
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:, 0:dim]
    b = hull.equations[:, dim]
    return A, -b, hull  # Negative moves b to the RHS of the inequality


def inner_ellipsoid_fit(points: ArrayLike):
    """
    Find the inscribed ellipsoid into a set of points of maximum volume. Return its matrix-offset form.

    original author: Raluca Sandu
    license: MIT
    reference: https://github.com/rmsandu/Ellipsoid-Fit

    """
    dim = points.shape[1]
    A, b, _ = GetHull(points)

    B = cp.Variable((dim, dim), PSD=True)  # Ellipsoid
    d = cp.Variable(dim)  # Center

    constraints = [
        cp.norm(B @ A[i], 2) + A[i] @ d
        <= b[i] + 5e-1  # additional tolerance so it fits boundaries
        for i in range(len(A))
    ]
    prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = prob.solve()
    if optval == np.inf:
        return None, None

    return B.value, d.value


def fit_ellipsoid(points: ArrayLike) -> Tuple[ArrayLike]:
    assert points.shape[1] == 3
    A, b = inner_ellipsoid_fit(points)
    if A is None:
        warnings.warn("Could not fit ellipsoid.")
        return tuple(points.T)

    invA = np.linalg.inv(A)

    centered = (points - b) @ invA
    inside = np.linalg.norm(centered, axis=1, ord=2) <= 1
    return tuple(points[inside].T)


@click.command()
@click.option(
    "--input-path",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input dataset path.",
)
@click.option(
    "--channel", "-c", required=True, type=str, help="Selected label channel."
)
@click.option(
    "--output-path", "-o", required=True, type=str, help="Output dataset path."
)
def cli_fit_ellipsoids(
    input_path: str,
    channel: str,
    output_path: str,
):
    in_ds = ZDataset(input_path)
    out_ds = ZDataset(output_path, mode="w", parent=in_ds)

    in_array = in_ds.get_array(channel)
    out_array = out_ds.add_channel(
        channel, in_array.shape, in_array.dtype, enable_projections=False
    )

    pbar = tqdm(range(in_array.shape[0]))
    for t in pbar:
        stack = np.zeros(out_array.shape[1:])
        regions = regionprops(in_array[t], cache=False)
        for i, region in enumerate(regions):
            ellipsoid_coords = fit_ellipsoid(region.coords)
            stack[ellipsoid_coords] = region.label
            pbar.set_postfix({"processing": f"{i}/{len(regions)}"})

        out_array[t] = stack

    in_ds.close()
    out_ds.close()
