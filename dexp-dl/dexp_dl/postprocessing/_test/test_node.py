import numpy as np

from dexp_dl.postprocessing.node import Node


def test_fast_find_binary_object():

    mask = np.zeros((124, 124, 124), dtype=bool)

    slicing = tuple(
        (
            slice(32, 75),
            slice(54, 65),
            slice(24, 100),
        )
    )
    mask[slicing] = True

    computed_slice = Node._fast_find_binary_object(mask)
    for a, b in zip(slicing, computed_slice):
        assert a.start == b.start and a.stop == b.stop
