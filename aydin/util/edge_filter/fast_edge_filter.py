import numba
import numpy
from numba import jit, stencil
from numpy.typing import ArrayLike


def fast_edge_filter(array: ArrayLike, axis: int = 0, gpu: bool = True):

    # Cast to float:
    array = array.astype(dtype=numpy.float32, copy=False)

    negative = ['0'] * array.ndim
    negative[axis] = '-1'
    positive = ['0'] * array.ndim
    positive[axis] = '+1'

    kernel_str = f"""
def kernel(array):
    return array[{','.join(positive)}] - array[{','.join(negative)}]

"""

    result = {}
    exec(kernel_str, result)

    _kernel = stencil(result['kernel'])

    def _fast_edge_filter(array: ArrayLike):
        array = _kernel(array)
        return array

    if gpu and numba.cuda.is_available():
        from numba import njit

        edge_func = njit(parallel=False)(_fast_edge_filter)
    else:
        edge_func = jit(parallel=True)(_fast_edge_filter)

    return edge_func(array)
