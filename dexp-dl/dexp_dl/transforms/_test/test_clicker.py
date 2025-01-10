import torch as th

from dexp_dl.transforms import Clicker


def test_automatic_click():

    target = th.zeros((100, 100), dtype=th.int32)
    target[40:61, 40:61] = 1

    clicker = Clicker(radius=3, target=target)
    clicker.query_next_map()
    click = clicker.clicks[-1]

    assert click.position == (50, 50) and click.is_positive

    pred = th.zeros_like(target)
    pred[40:61, 40:61] = 1
    pred[40:51, 40:51] = 0

    clicker.query_next_map(pred)
    click = clicker.clicks[-1]

    assert click.position == (45, 45) and click.is_positive

    pred[...] = 0
    pred[30:61, 30:61] = 1

    clicker.query_next_map(pred)
    click = clicker.clicks[-1]

    assert click.position == (35, 35) and not click.is_positive
