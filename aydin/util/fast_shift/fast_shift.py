from math import ceil
from typing import Tuple, Optional

import numba
import numpy
from numba import jit, prange

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def fast_shift(
    image, shift: Tuple[int], output=None, add=False, parallelism: Optional[int] = 8
):

    # Save original image dtype:
    original_dtype = image.dtype

    # Numba does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype
    image = image.astype(dtype=dtype, copy=False)

    # return type:
    output = numpy.zeros_like(image) if output is None else output

    parallelism = numba.get_num_threads() if parallelism is None else parallelism

    if image.ndim == 1:
        output = _fast_shift_1d(image, shift, output, add, parallelism)
    elif image.ndim == 2:
        output = _fast_shift_2d(image, shift, output, add, parallelism)
    elif image.ndim == 3:
        output = _fast_shift_3d(image, shift, output, add, parallelism)
    elif image.ndim == 4:
        output = _fast_shift_4d(image, shift, output, add, parallelism)
    else:
        raise ValueError("Image dimensions must be less or equal to 4.")

    # Make sure that output image has correct dtype:
    output = output.astype(dtype=original_dtype, copy=False)

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _fast_shift_1d(image, shift: Tuple[int], output, add: bool, parallelism: int):

    width = image.shape[0]
    chunk_width = int(ceil(width / parallelism))
    shift_x = shift[0]

    for c in prange(parallelism):
        for k in range(chunk_width):
            src_x = k + c * chunk_width
            dst_x = src_x + shift_x
            if 0 <= dst_x < width and src_x < width:
                if add:
                    output[dst_x] += image[src_x]
                else:
                    output[dst_x] = image[src_x]

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _fast_shift_2d(image, shift: Tuple[int], output, add: bool, parallelism: int):

    height = image.shape[0]
    width = image.shape[1]
    chunk_height = int(ceil(height / parallelism))
    shift_y = shift[0]
    shift_x = shift[1]

    for c in prange(parallelism):
        for k in range(chunk_height):
            src_y = k + c * chunk_height
            dst_y = src_y + shift_y
            if 0 <= dst_y < height and src_y < height:
                for src_x in range(width):
                    dst_x = src_x + shift_x
                    if 0 <= dst_x < width:
                        if add:
                            output[dst_y, dst_x] += image[src_y, src_x]
                        else:
                            output[dst_y, dst_x] = image[src_y, src_x]
    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _fast_shift_3d(image, shift: Tuple[int], output, add: bool, parallelism: int):

    depth = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    shift_z = shift[0]
    shift_y = shift[1]
    shift_x = shift[2]
    chunk_depth = int(ceil(depth / parallelism))

    for c in prange(parallelism):
        for k in range(chunk_depth):
            src_z = k + c * chunk_depth
            dst_z = src_z + shift_z
            if 0 <= dst_z < depth and src_z < depth:
                for src_y in range(height):
                    dst_y = src_y + shift_y
                    if 0 <= dst_y < height:
                        for src_x in range(width):
                            dst_x = src_x + shift_x
                            if 0 <= dst_x < width:
                                if add:
                                    output[dst_z, dst_y, dst_x] += image[
                                        src_z, src_y, src_x
                                    ]
                                else:
                                    output[dst_z, dst_y, dst_x] = image[
                                        src_z, src_y, src_x
                                    ]
    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _fast_shift_4d(image, shift: Tuple[int], output, add: bool, parallelism: int):

    length = image.shape[0]
    depth = image.shape[1]
    height = image.shape[2]
    width = image.shape[3]
    shift_w = shift[0]
    shift_z = shift[1]
    shift_y = shift[2]
    shift_x = shift[3]
    chunk_length = int(ceil(length / parallelism))

    for c in prange(parallelism):
        for k in range(chunk_length):
            src_w = k + c * chunk_length
            dst_w = src_w + shift_w
            if 0 <= dst_w < length and src_w < length:
                for src_z in range(depth):
                    dst_z = src_z + shift_z
                    if 0 <= dst_z < depth:
                        for src_y in range(height):
                            dst_y = src_y + shift_y
                            if 0 <= dst_y < height:
                                for src_x in range(width):
                                    dst_x = src_x + shift_x
                                    if 0 <= dst_x < width:
                                        if add:
                                            output[dst_w, dst_z, dst_y, dst_x] += image[
                                                src_w, src_z, src_y, src_x
                                            ]
                                        else:
                                            output[dst_w, dst_z, dst_y, dst_x] = image[
                                                src_w, src_z, src_y, src_x
                                            ]
    return output
