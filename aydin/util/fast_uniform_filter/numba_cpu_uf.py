from math import ceil

import numba
import numpy
from numba import jit, prange

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def numba_cpu_uniform_filter(
    image, size=3, output=None, mode="nearest", cval=0.0, origin=0
):
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

            # lprint(f"axis: {axis}, filter_size: {filter_size}")

            # set the parallelism:
            parallelism = numba.get_num_threads()
            # lprint(f"Number of threads: {parallelism}")
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
    """
    Numba jitted and parallelized method to apply uniform filter across
    last axis of the image provided. Doesn't return anything, output array
    should be provided as an argument.

    Parameters
    ----------
    image
    output
    filter_size
    parallelism

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
    """
    Numba jitted and parallelized method to apply uniform filter across
    last axis of the image provided. Doesn't return anything, output array
    should be provided as an argument.

    Parameters
    ----------
    image
    output
    filter_size

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
    """
    Numba jitted and parallelized method to apply uniform filter across
    last axis of the image provided. Doesn't return anything, output array
    should be provided as an argument.

    Parameters
    ----------
    image
    output
    filter_size

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
