from typing import Tuple

import higra as hg
import numba as nb
import numpy as np
from numpy.typing import ArrayLike


@nb.njit()
def edges_from_mask(
    mask: ArrayLike, image: ArrayLike, anisotropy_pen: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:

    reference = np.empty(mask.shape, dtype=np.int64)

    count = 0
    for z in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[2]):
                if mask[z, y, x]:
                    reference[z, y, x] = count
                    count += 1

    sources = np.empty(count * 3, dtype=np.int64)
    targets = np.empty(count * 3, dtype=np.int64)
    weights = np.empty(count * 3, dtype=np.float32)

    # not allowed to be annotated yet
    def update_fun(p, q, index, penalization=0.0):
        sources[index] = reference[p]
        targets[index] = reference[q]
        weights[index] = (image[p] + image[q]) / 2.0 + 1e-8 + penalization
        # avoid zero weight, could lead to problems with sparse matrices

    count = 0
    for z in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[2]):
                cur = (z, y, x)
                if mask[cur]:
                    if x + 1 < mask.shape[2]:
                        neigh = (z, y, x + 1)
                        if mask[neigh]:
                            update_fun(cur, neigh, count)
                            count += 1

                    if y + 1 < mask.shape[1]:
                        neigh = (z, y + 1, x)
                        if mask[neigh]:
                            update_fun(cur, neigh, count)
                            count += 1

                    if z + 1 < mask.shape[0]:
                        neigh = (z + 1, y, x)
                        if mask[neigh]:
                            update_fun(cur, neigh, count, anisotropy_pen)
                            count += 1

    return sources[:count], targets[:count], weights[:count]


def mask_to_graph(
    mask: ArrayLike, image: ArrayLike, anisotropy_pen: float
) -> Tuple[hg.UndirectedGraph, ArrayLike]:
    assert mask.shape == image.shape
    assert mask.dtype == bool, f"Found {mask.dtype}."

    sources, targets, weights = edges_from_mask(mask, image, anisotropy_pen)

    graph = hg.UndirectedGraph()
    graph.add_vertices(mask.sum())
    graph.add_edges(sources, targets)

    hg.set_attribute(graph, "no_border_vertex_out_degree", 2 * mask.ndim)

    return graph, weights


def image_3d_to_graph(image: ArrayLike) -> hg.UndirectedGraph:
    assert image.ndim == 3

    neighbors = []
    for i in range(3):

        for j in (1, -1):
            shift = [0] * 3
            shift[i] = j
            neighbors.append(shift)

    return hg.get_nd_regular_graph(image.shape, neighbors)


def image_3d_to_hierarchy(image: ArrayLike) -> Tuple[hg.Tree, ArrayLike]:
    graph = image_3d_to_graph(image)
    weights = hg.weight_graph(graph, image, hg.WeightFunction.L1)
    return hg.watershed_hierarchy_by_volume(graph, weights)
