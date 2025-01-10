from typing import Optional, Tuple, Sequence

import click
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from dexp.datasets import ZDataset
from dexp.processing.restoration import dehazing, lipshitz_correction
from dexp.utils.backends import CupyBackend
from dexp.cli.parsing import device_option, output_dataset_options, input_dataset_argument, channels_option
from scipy.ndimage import median_filter


def gray_normalize(
    image: ArrayLike,
    label: ArrayLike,
    epsilon: float = 1e-8,
    lower: float = 0.005,
    upper: float = 0.999,
    gamma: Optional[float] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[ArrayLike, ArrayLike]:

    if isinstance(image, cp.ndarray):
        xp = cp
    else:
        xp = np

    if gamma is not None:
        image = xp.power(image, gamma)

    # reducing memory usage
    if image.size > 128 ** 3:
        q_input = image[::4, ::4, ::4]
    else:
        q_input = image

    lb = xp.quantile(q_input, lower).item()
    up = xp.quantile(q_input - lb, upper).item()

    image = xp.clip(image - lb, 0, up)
    image = image.astype(dtype) / (up + epsilon)

    return image, label


@click.command()
@input_dataset_argument()
@output_dataset_options()
@device_option()
@channels_option()
@click.option(
    "--lower",
    "-l",
    type=float,
    default=0.005,
    help="Lower quantile of normalization",
    show_default=True,
)
@click.option(
    "--upper",
    "-u",
    type=float,
    default=0.999,
    help="Upper quantile of normalization",
    show_default=True,
)
@click.option(
    "--median-window",
    "-m",
    type=int,
    default=0,
    help="Smooth thresholds with a median filtering given the window size",
    show_default=True,
)
@click.option(
    "--dehaze/--no-dehaze",
    "-dh/-ndh",
    is_flag=True,
    type=bool,
    default=False,
    help="Dehaze and correct continuity before normalizing data.",
)
@click.option(
    "--gamma",
    "-g",
    type=float,
    default=None,
    help="Exponent for gamma correction. I ^ gamma.",
)
def cli_gray_normalize(
    input_dataset: ZDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    upper: float,
    lower: float,
    median_window: int,
    device: int,
    dehaze: bool,
    gamma: Optional[float],
):
    dtype = np.float16
    
    for channel in channels:
        in_arr = input_dataset.get_array(channel)
        out_arr = output_dataset.add_channel(channel, shape=in_arr.shape, dtype=dtype)

        with CupyBackend(device_id=device) as backend:
            if dehaze:
                for t in tqdm(range(in_arr.shape[0]), "Correcting continuity and dehazing"):
                    cu_arr = backend.to_backend(in_arr[t])
                    cu_arr = lipshitz_correction.lipschitz_continuity_correction(
                        cu_arr, internal_dtype=np.float32
                    )
                    out_arr[t] = backend.to_numpy(
                        dehazing.dehaze(cu_arr, internal_dtype=np.float32)
                    )

                # input for normalization is in out_arr now
                in_arr = out_arr

            if median_window == 0:
                for t in tqdm(range(in_arr.shape[0])):
                    out, _ = gray_normalize(
                        backend.to_backend(in_arr[t]),
                        label=None,
                        lower=lower,
                        upper=upper,
                        gamma=gamma,
                        dtype=dtype,
                    )
                    out_arr[t] = backend.to_numpy(out)
            else:
                lowers = []
                uppers = []
                for t in tqdm(range(in_arr.shape[0]), "Finding thresholds"):
                    cu_arr = backend.to_backend(in_arr[t])
                    lowers.append(cp.quantile(cu_arr, lower).item())
                    uppers.append(cp.quantile(cu_arr, upper).item())

                lowers = median_filter(np.array(lowers), size=median_window, mode="nearest")
                uppers = median_filter(np.array(uppers), size=median_window, mode="nearest")
                if gamma is not None:
                    lowers = lowers**gamma
                    uppers = uppers**gamma

                for t in tqdm(range(in_arr.shape[0]), "Normalizing"):
                    cu_arr = backend.to_backend(in_arr[t])
                    if gamma is not None:
                        cu_arr = cp.power(cu_arr, gamma)
                    cu_arr = cu_arr - lowers[t]
                    u = uppers[t] - lowers[t]
                    cu_arr = cp.clip(cu_arr, 0, u)
                    cu_arr = cu_arr.astype(dtype) / (u + 1e-8)
                    out_arr[t] = backend.to_numpy(cu_arr)

        input_dataset.close()
        output_dataset.close()
