import logging
from pathlib import Path
from typing import Callable, Sequence, Tuple, cast

import click
import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
import sqlalchemy as sqla
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len
from sqlalchemy.orm import Session
from ultrack.cli.utils import config_option, napari_reader_option, paths_argument
from ultrack.config import MainConfig
from ultrack.core.database import NodeDB

from utils import center_crop, pad_to_shape, to_cpu, unified_memory

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.0,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
) -> Tuple[int, ...]:
    """
    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    LOG.info(
        f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
        f"with maximum shift of {maximum_shift}"
    )

    if np.any(shape > ref_img.shape):
        padded_shape = np.maximum(ref_img.shape, shape)
        ref_img = pad_to_shape(ref_img, padded_shape, mode="reflect")
        mov_img = pad_to_shape(mov_img, padded_shape, mode="reflect")

    if np.any(shape < ref_img.shape):
        ref_img = center_crop(ref_img, shape)
        mov_img = center_crop(mov_img, shape)

    ref_img = to_device(ref_img)
    mov_img = to_device(mov_img)

    # ref_img = np.log1p(ref_img)
    # mov_img = np.log1p(mov_img)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()
    del Fimg1, Fimg2

    norm = np.fmax(np.abs(prod), eps)
    corr = np.fft.irfftn(prod / norm)
    del prod, norm

    corr = np.fft.fftshift(np.abs(corr))

    peak = np.unravel_index(to_cpu(np.argmax(corr)), corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    LOG.info(f"phase cross corr. peak at {peak}")

    return peak


def register(config: MainConfig, arr: ArrayLike) -> None:

    with unified_memory():
        cur_img = cp.asarray(arr[0])
        cur_img = ndi.zoom(cur_img, order=1, zoom=(0.5,) * 3)

        assert cur_img.ndim == 3, f"Expected 3D image, got {cur_img.shape}"

        engine = sqla.create_engine(config.data_config.database_path)

        for t in range(1, arr.shape[0]):
            prev_img = cur_img
            cur_img = cp.asarray(arr[t])
            cur_img = ndi.zoom(cur_img, order=1, zoom=(0.5,) * 3)

            shift = phase_cross_corr(
                ref_img=cur_img,
                mov_img=prev_img,
            )
            shift = tuple(s * 2.0 for s in shift)
            print(t, shift)

            with Session(engine) as session:
                statement = (
                    sqla.update(NodeDB)
                    .where(NodeDB.t == t)
                    .values(
                        z_shift=shift[0],
                        y_shift=shift[1],
                        x_shift=shift[2],
                    )
                )
                session.execute(
                    statement,
                    execution_options={"synchronize_session": False},
                )
                session.commit()


@click.command()
@paths_argument()
@napari_reader_option()
@config_option()
def main(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
) -> None:
    """Adds coordinates shift to segmentation hypotheses."""
    _initialize_plugins()

    viewer = ViewerModel()

    layer = viewer.open(paths, plugin=reader_plugin)[0]
    image = layer.data[0] if layer.multiscale else layer.data

    register(
        config,
        image,
    )


if __name__ == "__main__":
    main()
