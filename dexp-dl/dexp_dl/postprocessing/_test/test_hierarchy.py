import higra as hg
import numpy as np
from skimage import data, filters, segmentation

from dexp_dl.postprocessing.hierarchy import create_hierarchies, to_labels


def test_connected_comp_hierarchy(display: bool = False):

    mask = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.25)

    boundaries = segmentation.find_boundaries(mask)
    boundaries = filters.gaussian(boundaries, sigma=2)
    boundaries += np.random.rand(*boundaries.shape) * 0.1

    hierarchies = create_hierarchies(mask, boundaries, hg.watershed_hierarchy_by_area)

    connected_comp = to_labels(hierarchies, mask.shape)
    for h in hierarchies:
        h.cut_threshold = 25
    threshold_segms = to_labels(hierarchies, mask.shape)

    if display:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(mask, name="Mask", blending="additive")
        viewer.add_image(boundaries, name="Weights", blending="additive")
        viewer.add_labels(connected_comp, name="Connec. Comp.")
        viewer.add_labels(threshold_segms, name="Segments")

        napari.run()


if __name__ == "__main__":
    test_connected_comp_hierarchy(True)
