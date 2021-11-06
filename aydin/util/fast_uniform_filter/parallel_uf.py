import multiprocessing

import numpy
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter

from aydin.util.array.nd import nd_split_slices, remove_margin_slice


def parallel_uniform_filter(
    image, size=3, output=None, mode="nearest", cval=0.0, origin=0
):
    # Save original image dtype:
    original_dtype = image.dtype

    # Scipy does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype
    image = image.astype(dtype=dtype, copy=False)

    # Number of parallel jobs:
    num_jobs = max(1, int(0.95 * multiprocessing.cpu_count()))

    # TODO: adjust the number of jobs to the filter size

    # Longest axis:
    longest_axis_length = max(image.shape)

    # pick the longest axis for splitting:
    longest_axis = list(image.shape).index(longest_axis_length)

    # Normalise filter size to tuple:
    size = size if isinstance(size, tuple) else (size,) * image.ndim

    # If the filter size is too large, there is no point to split:
    filter_size_along_longest_axis = size[longest_axis]
    if (
        longest_axis_length // num_jobs + filter_size_along_longest_axis
        < longest_axis_length * 0.9  # we need to gain at least 10% speed!
    ):
        # No point in going parallel, we won't gain anything:
        output = uniform_filter(image, size=size, mode=mode, cval=cval, origin=origin)
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

        # Instantiates output array:
        output = numpy.empty_like(image)

        def _uniform_filter(slice_tuple, slice_margin_tuple):
            tile = image[slice_margin_tuple]

            output_tile = uniform_filter(
                tile, size=size, mode=mode, cval=cval, origin=origin
            )

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
            Parallel()(delayed(_uniform_filter)(st, smt) for st, smt in slices)

    output = output.astype(dtype=original_dtype, copy=False)

    return output
