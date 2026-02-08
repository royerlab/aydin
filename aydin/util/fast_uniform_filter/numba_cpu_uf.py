"""Numba-accelerated CPU uniform (box) filter for n-dimensional arrays.

Implements a separable uniform filter using a sliding-window accumulator
approach for each axis, parallelized across slices using Numba JIT.
"""

from math import ceil

import numba
import numpy
from numba import jit, prange

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def numba_cpu_uniform_filter(
    image, size=3, output=None, mode="nearest", cval=0.0, origin=0
):
    """Apply a uniform (box) filter using Numba-accelerated separable filtering.

    Applies the filter axis-by-axis using an efficient sliding-window
    accumulator. Supports arrays up to 4D.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array (up to 4D).
    size : int or tuple of int
        Filter window size. If a tuple, specifies size per axis.
    output : numpy.ndarray, optional
        Not used directly; kept for API compatibility.
    mode : str
        Boundary mode. Only 'nearest' is supported.
    cval : float
        Not used (kept for API compatibility).
    origin : int
        Not used (kept for API compatibility).

    Returns
    -------
    numpy.ndarray
        Filtered image with the original dtype.
    """
    # Save original image dtype:
    original_dtype = image.dtype

    # Numba does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype
    image = image.astype(dtype=dtype, copy=False)

    # Instantiates working images:
    image_a_0 = numpy.empty(image.shape, dtype=dtype)
    image_b_0 = numpy.empty(image.shape, dtype=dtype)

    # Current output (just a ref)
    output = None

    axes = list(range(image.ndim))
    if len(axes) == 1:
        _cpu_line_filter(image, image_a_0, size)
        output = image_a_0
    elif len(axes) > 0:
        for axis in axes:

            if axis == 0:
                image_a = image
                image_b = image_a_0

            filter_size = size[axis] if isinstance(size, tuple) else size

            # aprint(f"axis: {axis}, filter_size: {filter_size}")

            # set the parallelism:
            parallelism = numba.get_num_threads()
            # aprint(f"Number of threads: {parallelism}")
            # max(1, int(0.9*multiprocessing.cpu_count()))

            uniform_filter1d_with_conditionals(
                image_a, image_b, filter_size, axis, parallelism=parallelism
            )

            output = image_b

            if axis == 0:
                image_a = image_a_0
                image_b = image_b_0
            else:
                image_b, image_a = image_a, image_b
    else:
        output = image.copy()

    # Make sure that output image has correct dtype:
    output = output.astype(dtype=original_dtype, copy=False)

    return output


def uniform_filter1d_with_conditionals(image, output, filter_size, axis, parallelism=8):
    """Apply 1D uniform filter along a specific axis with axis swapping.

    Dispatches to the appropriate Numba kernel based on the number
    of array dimensions, swapping axes as needed so the filter always
    operates along the last axis.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    output : numpy.ndarray
        Output array (must be pre-allocated, same shape as image).
    filter_size : int
        Window size for the uniform filter.
    axis : int
        Axis along which to apply the filter.
    parallelism : int
        Number of parallel threads.
    """
    if image.ndim == 1:
        _cpu_line_uniform_filter_without_loops(image, output, filter_size, parallelism)
    elif image.ndim == 2:
        prepared_image = image.swapaxes(0, 1) if axis == 0 else image
        prepared_output = output.swapaxes(0, 1) if axis == 0 else output
        _cpu_line_uniform_filter_without_loops(
            prepared_image, prepared_output, filter_size, parallelism
        )
    elif image.ndim == 3:
        prepared_image = image.swapaxes(axis, 2) if axis != 2 else image
        prepared_output = output.swapaxes(axis, 2) if axis != 2 else output
        _cpu_line_uniform_filter_with_2d_loop(
            prepared_image, prepared_output, filter_size, parallelism
        )
        # cpu_line_uniform_filter_with_2d_loop.parallel_diagnostics(level=4)
        # for key, value in cpu_line_filter.inspect_asm().items():
        #     print(f"{key} -> {value}")

    elif image.ndim == 4:
        prepared_image = image.swapaxes(axis, 3) if axis != 3 else image
        prepared_output = output.swapaxes(axis, 3) if axis != 3 else output
        _cpu_line_uniform_filter_with_3d_loop(
            prepared_image, prepared_output, filter_size, parallelism
        )
        # cpu_line_uniform_filter_with_3d_loop.parallel_diagnostics(level=4)


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _cpu_line_uniform_filter_without_loops(image, output, filter_size, parallelism=8):
    """Apply uniform filter along the last axis of a 2D array in parallel.

    Parameters
    ----------
    image : numpy.ndarray
        2D input array.
    output : numpy.ndarray
        2D output array (pre-allocated, same shape as image).
    filter_size : int
        Window size for the uniform filter.
    parallelism : int
        Number of parallel chunks to process.
    """

    length = image.shape[0]
    chunk_length = int(ceil(length / parallelism))

    for c in prange(parallelism):
        for k in range(chunk_length):
            i = k + c * chunk_length
            if i < length:
                input_line = image[i, :]
                output_line = output[i, :]
                _cpu_line_filter(input_line, output_line, filter_size)

    # print(cpu_line_filter.inspect_llvm())


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _cpu_line_uniform_filter_with_2d_loop(image, output, filter_size, parallelism=8):
    """Apply uniform filter along the last axis of a 3D array in parallel.

    Parameters
    ----------
    image : numpy.ndarray
        3D input array.
    output : numpy.ndarray
        3D output array (pre-allocated, same shape as image).
    filter_size : int
        Window size for the uniform filter.
    parallelism : int
        Number of parallel chunks to process.
    """

    height = image.shape[0]
    width = image.shape[1]
    chunk_height = int(ceil(height / parallelism))

    for c in prange(parallelism):
        for k in range(chunk_height):
            y = k + c * chunk_height
            image_y = image[y]
            output_y = output[y]
            if y < height:
                for x in range(width):
                    input_line = image_y[x]
                    output_line = output_y[x]
                    _cpu_line_filter(input_line, output_line, filter_size)


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _cpu_line_uniform_filter_with_3d_loop(image, output, filter_size, parallelism=8):
    """Apply uniform filter along the last axis of a 4D array in parallel.

    Parameters
    ----------
    image : numpy.ndarray
        4D input array.
    output : numpy.ndarray
        4D output array (pre-allocated, same shape as image).
    filter_size : int
        Window size for the uniform filter.
    parallelism : int
        Number of parallel chunks to process.
    """

    depth = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    chunk_depth = int(ceil(depth / parallelism))

    for c in prange(parallelism):
        for k in range(chunk_depth):
            z = k + c * chunk_depth
            if z < depth:
                image_z = image[z]
                output_z = output[z]
                for y in range(height):
                    image_y = image_z[y]
                    output_y = output_z[y]
                    for x in range(width):
                        input_line = image_y[x]
                        output_line = output_y[x]
                        _cpu_line_filter(input_line, output_line, filter_size)


@jit(nopython=True, error_model=__error_model, fastmath=__fastmath)
def _cpu_line_filter(input_line, output_line, filter_size):
    """
    Numba jitted line filter implementation. Doesn't return anything,
    output array should be provided as an argument.

    Parameters
    ----------
    input_line
        1D input array
    output_line
        1D output array
    filter_size
        Size of the uniform filter
    """

    def safe_index(index, size):
        if index < 0:
            return 0
        elif index >= size:
            return size - 1
        else:
            return index

    array_size = len(input_line)
    left_offset, right_offset = (filter_size // 2, filter_size - (filter_size // 2) - 1)

    tmp = numpy.float64(0.0)

    for ind in range(-left_offset, right_offset + 1):
        tmp += input_line[safe_index(ind, array_size)]

    output_line[0] = tmp / filter_size

    for ind in range(1, array_size):
        element_to_add_index = safe_index(ind + right_offset, array_size)
        element_to_sub_index = safe_index(ind - left_offset - 1, array_size)

        tmp += input_line[element_to_add_index]
        tmp -= input_line[element_to_sub_index]
        output_line[ind] = tmp / filter_size
