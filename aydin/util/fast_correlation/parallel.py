import multiprocessing

import numpy
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.ndimage import correlate

from aydin.util.array.nd import nd_split_slices, remove_margin_slice


def parallel_correlate(
    image: ArrayLike,
    kernel: ArrayLike,
    output: ArrayLike = None,
    cpu_load: float = 0.95,
):

    # Save original image dtype:
    original_dtype = image.dtype

    # Numba does not support float16 yet:
    dtype = numpy.float32
    image = image.astype(dtype=dtype, copy=False)
    kernel = kernel.astype(dtype=dtype, copy=False)

    # Instantiates output array:
    if output is None:
        output = numpy.empty_like(image)

    # Number of parallel jobs:
    num_jobs = max(1, int(cpu_load * multiprocessing.cpu_count()))

    # Longest axis:
    longest_axis_length = max(image.shape)

    # pick the longest axis for splitting:
    longest_axis = list(image.shape).index(longest_axis_length)

    # Kernel size:
    size = kernel.shape

    # If the filter size is too large, there is no point to split:
    filter_size_along_longest_axis = size[longest_axis]
    if (
        longest_axis_length // num_jobs + filter_size_along_longest_axis
        < longest_axis_length * 0.9  # we need to gain at least 10% speed!
    ):
        # No point in going parallel, we won't gain anything:
        output = correlate(image, weights=kernel, output=output)
    else:
        # configure splitting:
        nb_slices = [1] * image.ndim
        nb_slices[longest_axis] = min(num_jobs, image.shape[longest_axis])
        nb_slices = tuple(nb_slices)
        margins = (
            (size[longest_axis] // 2,) * image.ndim
            if isinstance(size, tuple)
            else (size // 2,) * image.ndim
        )

        # Obtain slice objects for splitting:
        slice_tuple_list = list(
            nd_split_slices(image.shape, nb_slices, do_shuffle=False)
        )
        slice_margin_tuple_list = list(
            nd_split_slices(image.shape, nb_slices, do_shuffle=False, margins=margins)
        )

        def _correlate(slice_tuple, slice_margin_tuple):
            tile = image[slice_margin_tuple]

            output_tile = correlate(tile, weights=kernel)

            remove_margin_slice_tuple = remove_margin_slice(
                image.shape, slice_margin_tuple, slice_tuple
            )
            output_tile_without_margin = output_tile[remove_margin_slice_tuple]

            output[slice_tuple] = output_tile_without_margin

        slices = (
            (st, smt) for st, smt in zip(slice_tuple_list, slice_margin_tuple_list)
        )

        from joblib import parallel_backend

        with parallel_backend('threading', n_jobs=num_jobs):
            Parallel()(delayed(_correlate)(st, smt) for st, smt in slices)

    output[...] = output.astype(dtype=original_dtype, copy=False)

    return output
