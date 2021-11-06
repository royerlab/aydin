import math

import numpy
from numba import get_num_threads, cuda
from numba.cuda import device_array

from aydin.util.fast_uniform_filter.numba_cpu_uf import _cpu_line_filter


def numba_gpu_uniform_filter(
    image, size=3, cuda_stream=0, output=None, mode="nearest", cval=0.0, origin=0
):

    original_dtype = image.dtype

    # Numba does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype

    # Instantiates working images:
    image_a_0 = device_array(image.shape, dtype=dtype, stream=cuda_stream)
    image_b_0 = device_array(image.shape, dtype=dtype, stream=cuda_stream)

    # Current output (just a ref)
    output = None

    axes = list(range(image.ndim))
    if len(axes) == 1:
        # Not worth using the gpu for that...
        _cpu_line_filter(image, image_a_0, size)
        output = image_a_0
    elif len(axes) > 0:
        for axis in axes:

            if axis == 0:
                image_a = image
                image_b = image_a_0

            filter_size = size[axis] if isinstance(size, tuple) else size

            # set the parallelism:
            parallelism = get_num_threads()

            # Currently using conditional implementation, can switch to reshape or slicing
            uniform_filter1d_with_conditionals_gpu(
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
    array = numpy.empty(shape=output.shape, dtype=dtype)
    output.copy_to_host(array, stream=cuda_stream)

    return array


def uniform_filter1d_with_conditionals_gpu(
    image, output, current_size, axis, parallelism=None
):
    if image.ndim == 1:
        raise Exception("numba gpu implementation is no supporting 1D input arrays.")
    elif image.ndim == 2:
        axes = list(range(image.ndim))
        axes.remove(axis)
        first_axis_index = axes[0]
        length1 = numpy.prod(image.shape[first_axis_index + 1 :])
        step_size = numpy.prod(image.shape[axis + 1 :])

        threadsperblock = 32 * 32
        blockspergrid = int(math.ceil(image.shape[first_axis_index] / threadsperblock))

        # In some cases if the image is too small, better have more blocks and less threads per block:
        if blockspergrid < 256:
            threadsperblock = 1
            blockspergrid = int(
                math.ceil(image.shape[first_axis_index] / threadsperblock)
            )

        gpu_line_filter_2d[blockspergrid, threadsperblock](
            image.ravel(),
            output.ravel(),
            current_size,
            image.shape[axis],
            length1,
            step_size,
            image.shape[first_axis_index],
        )
    elif image.ndim == 3:
        axes = list(range(image.ndim))
        axes.remove(axis)
        first_axis_index, second_axis_index = tuple(axes)
        length1 = numpy.prod(image.shape[first_axis_index + 1 :])
        length2 = numpy.prod(image.shape[second_axis_index + 1 :])
        step_size = numpy.prod(image.shape[axis + 1 :])

        threadsperblock = (32, 32)
        blockspergrid_x = int(
            math.ceil(image.shape[first_axis_index] / threadsperblock[0])
        )
        blockspergrid_y = int(
            math.ceil(image.shape[second_axis_index] / threadsperblock[1])
        )
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # In some cases if the image is too small, better have more blocks and less threads per block:
        if blockspergrid_x < 256 or blockspergrid_y < 256:
            threadsperblock = (
                1 if blockspergrid_x < 256 else 32,
                1 if blockspergrid_y < 256 else 32,
            )

            blockspergrid_x = int(
                math.ceil(image.shape[first_axis_index] / threadsperblock[0])
            )
            blockspergrid_y = int(
                math.ceil(image.shape[second_axis_index] / threadsperblock[1])
            )
            blockspergrid = (blockspergrid_x, blockspergrid_y)

        gpu_line_filter_3d[blockspergrid, threadsperblock](
            image.ravel(),
            output.ravel(),
            current_size,
            image.shape[axis],
            length1,
            length2,
            step_size,
            image.shape[first_axis_index],
            image.shape[second_axis_index],
        )

    elif image.ndim == 4:
        axes = list(range(image.ndim))
        axes.remove(axis)
        first_axis_index, second_axis_index, third_axis_index = tuple(axes)
        length1 = numpy.prod(image.shape[first_axis_index + 1 :])
        length2 = numpy.prod(image.shape[second_axis_index + 1 :])
        length3 = numpy.prod(image.shape[third_axis_index + 1 :])
        step_size = numpy.prod(image.shape[axis + 1 :])

        threadsperblock = (8, 8, 8)
        blockspergrid_x = int(
            math.ceil(image.shape[first_axis_index] / threadsperblock[0])
        )
        blockspergrid_y = int(
            math.ceil(image.shape[second_axis_index] / threadsperblock[1])
        )
        blockspergrid_z = int(
            math.ceil(image.shape[third_axis_index] / threadsperblock[2])
        )
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        gpu_line_filter_4d[blockspergrid, threadsperblock](
            image.ravel(),
            output.ravel(),
            current_size,
            image.shape[axis],
            length1,
            length2,
            length3,
            step_size,
            image.shape[first_axis_index],
            image.shape[second_axis_index],
            image.shape[third_axis_index],
        )


@cuda.jit
def gpu_line_filter_2d(
    image, output, filter_size, line_length, length1, step_size, x_length
):
    """
    Numba CUDA line filter implementation. Doesn't return anything,
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
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos >= x_length:
        return

    index = pos * length1

    input_line = image[index : index + (line_length * step_size) : step_size]
    output_line = output[index : index + (line_length * step_size) : step_size]
    gpu_line_filter(input_line, output_line, filter_size)


@cuda.jit
def gpu_line_filter_3d(
    image,
    output,
    filter_size,
    line_length,
    length1,
    length2,
    step_size,
    x_length,
    y_length,
):
    """
    Numba CUDA line filter implementation. Doesn't return anything,
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
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos_x = tx + ty * bw

    # Thread id in a 1D block
    tx = cuda.threadIdx.y
    # Block id in a 1D grid
    ty = cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.y
    # Compute flattened index inside the array
    pos_y = tx + ty * bw

    if pos_x >= x_length or pos_y >= y_length:
        return

    index = pos_x * length1 + pos_y * length2

    input_line = image[index : index + (line_length * step_size) : step_size]
    output_line = output[index : index + (line_length * step_size) : step_size]
    gpu_line_filter(input_line, output_line, filter_size)


@cuda.jit
def gpu_line_filter_4d(
    image,
    output,
    filter_size,
    line_length,
    length1,
    length2,
    length3,
    step_size,
    x_length,
    y_length,
    z_length,
):
    """
    Numba CUDA line filter implementation. Doesn't return anything,
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
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos_x = tx + ty * bw

    # Thread id in a 1D block
    tx = cuda.threadIdx.y
    # Block id in a 1D grid
    ty = cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.y
    # Compute flattened index inside the array
    pos_y = tx + ty * bw

    # Thread id in a 1D block
    tx = cuda.threadIdx.z
    # Block id in a 1D grid
    ty = cuda.blockIdx.z
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.z
    # Compute flattened index inside the array
    pos_z = tx + ty * bw

    if pos_x >= x_length or pos_y >= y_length or pos_z >= z_length:
        return

    index = pos_x * length1 + pos_y * length2 + pos_z * length3

    input_line = image[index : index + (line_length * step_size) : step_size]
    output_line = output[index : index + (line_length * step_size) : step_size]
    gpu_line_filter(input_line, output_line, filter_size)


@cuda.jit(device=True)
def gpu_line_filter(input_line, output_line, current_size):
    array_size = len(input_line)
    left_offset, right_offset = (
        current_size // 2,
        current_size - (current_size // 2) - 1,
    )

    def safe_index(index):
        if index < 0:
            return 0
        elif index >= array_size:
            return array_size - 1
        else:
            return index

    tmp = 0.0
    for ind in range(-left_offset, right_offset + 1):
        tmp += input_line[safe_index(ind)]

    output_line[0] = tmp / current_size

    for ind in range(1, array_size):
        element_to_add_index = safe_index(ind + right_offset)
        element_to_sub_index = safe_index(ind - left_offset - 1)

        tmp += input_line[element_to_add_index]
        tmp -= input_line[element_to_sub_index]
        output_line[ind] = tmp / current_size
