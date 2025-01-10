from typing import Callable
import click

from dexp_dl.transforms import dilate_edge_label


def _edge_dilation_callback(ctx: click.Context, opt: click.Option, value: int) -> None:
    output_transform = ctx.params["output_transform"]

    def _new_output_transform(image, label):
        image, label = output_transform(image, label)
        return dilate_edge_label(image, label, radius=value)

    ctx.params["output_transform"] = _new_output_transform


def edge_dilation_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--edge-dilation-radius",
            "-edr",
            default=0,
            type=int,
            show_default=True,
            expose_value=False,
            help="Dilate labels before comparison.",
            callback=_edge_dilation_callback,
        )(f)

    return decorator
