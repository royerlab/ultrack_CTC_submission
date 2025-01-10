from typing import Optional, Tuple

import cupy
import numpy as np
import zarr
from tqdm import tqdm


def gray_normalize(
    in_array: zarr.Array,
    epsilon: float = 1e-8,
    lower: float = 0.005,
    upper: float = 0.995,
    chunks: Optional[Tuple[int]] = None,
) -> zarr.Array:
    if chunks is None:
        chunks = in_array.chunks

    shape = in_array.shape
    out_array = zarr.empty(
        shape, store=zarr.TempStore(), dtype=np.float16, chunks=chunks
    )

    for i in tqdm(range(shape[0]), desc="Gray Normalizing"):
        stack = cupy.asarray(in_array[i].astype(float))
        lb = cupy.quantile(stack, lower)
        stack -= lb
        up = cupy.quantile(stack, upper)
        stack = cupy.clip(stack, 0, up)
        stack = stack / (up + epsilon)
        out_array[i] = stack.get()

    return out_array
