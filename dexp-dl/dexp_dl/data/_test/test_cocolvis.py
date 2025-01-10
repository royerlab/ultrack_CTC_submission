from pathlib import Path

import numpy as np
import pytest

from dexp_dl.data import CocoLvisDataset


def test_cocolvis_dataset(display: bool = False):
    ds_path = Path("/mnt/hd2/cocolvis")
    if not ds_path.exists():
        pytest.skip("COCO+LVIS dataset no found, skipping test.")

    dataset = CocoLvisDataset(ds_path, "train")
    import napari

    for i in range(5):
        image, mask = dataset[i]

        assert image.shape[:2] == mask.shape
        assert len(np.unique(mask) == 2)

        if display:
            viewer = napari.Viewer()
            viewer.add_image(image, rgb=True)
            viewer.add_labels(mask)

            napari.run()
