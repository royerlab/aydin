"""Fast edge detection filter using Numba-accelerated stencil operations."""

import numba
import numpy
from numba import jit, stencil
from numpy.typing import ArrayLike

# Pre-defined stencil functions for each dimension/axis combination.
# This avoids using exec() for dynamic code generation, improving security
# and allowing better static analysis of the code.


@stencil
def _kernel_2d_axis_0(array):
    """2D edge filter along axis 0."""
    return array[1, 0] - array[-1, 0]


@stencil
def _kernel_2d_axis_1(array):
    """2D edge filter along axis 1."""
    return array[0, 1] - array[0, -1]


@stencil
def _kernel_3d_axis_0(array):
    """3D edge filter along axis 0."""
    return array[1, 0, 0] - array[-1, 0, 0]


@stencil
def _kernel_3d_axis_1(array):
    """3D edge filter along axis 1."""
    return array[0, 1, 0] - array[0, -1, 0]


@stencil
def _kernel_3d_axis_2(array):
    """3D edge filter along axis 2."""
    return array[0, 0, 1] - array[0, 0, -1]


@stencil
def _kernel_4d_axis_0(array):
    """4D edge filter along axis 0."""
    return array[1, 0, 0, 0] - array[-1, 0, 0, 0]


@stencil
def _kernel_4d_axis_1(array):
    """4D edge filter along axis 1."""
    return array[0, 1, 0, 0] - array[0, -1, 0, 0]


@stencil
def _kernel_4d_axis_2(array):
    """4D edge filter along axis 2."""
    return array[0, 0, 1, 0] - array[0, 0, -1, 0]


@stencil
def _kernel_4d_axis_3(array):
    """4D edge filter along axis 3."""
    return array[0, 0, 0, 1] - array[0, 0, 0, -1]


@stencil
def _kernel_5d_axis_0(array):
    """5D edge filter along axis 0."""
    return array[1, 0, 0, 0, 0] - array[-1, 0, 0, 0, 0]


@stencil
def _kernel_5d_axis_1(array):
    """5D edge filter along axis 1."""
    return array[0, 1, 0, 0, 0] - array[0, -1, 0, 0, 0]


@stencil
def _kernel_5d_axis_2(array):
    """5D edge filter along axis 2."""
    return array[0, 0, 1, 0, 0] - array[0, 0, -1, 0, 0]


@stencil
def _kernel_5d_axis_3(array):
    """5D edge filter along axis 3."""
    return array[0, 0, 0, 1, 0] - array[0, 0, 0, -1, 0]


@stencil
def _kernel_5d_axis_4(array):
    """5D edge filter along axis 4."""
    return array[0, 0, 0, 0, 1] - array[0, 0, 0, 0, -1]


# Mapping of (ndim, axis) to pre-defined stencil function
_STENCIL_KERNELS = {
    (2, 0): _kernel_2d_axis_0,
    (2, 1): _kernel_2d_axis_1,
    (3, 0): _kernel_3d_axis_0,
    (3, 1): _kernel_3d_axis_1,
    (3, 2): _kernel_3d_axis_2,
    (4, 0): _kernel_4d_axis_0,
    (4, 1): _kernel_4d_axis_1,
    (4, 2): _kernel_4d_axis_2,
    (4, 3): _kernel_4d_axis_3,
    (5, 0): _kernel_5d_axis_0,
    (5, 1): _kernel_5d_axis_1,
    (5, 2): _kernel_5d_axis_2,
    (5, 3): _kernel_5d_axis_3,
    (5, 4): _kernel_5d_axis_4,
}


def fast_edge_filter(array: ArrayLike, axis: int = 0, gpu: bool = True):
    """Apply a fast edge detection filter along a specified axis.

    Computes the difference between neighboring pixels along the given axis
    using a Numba stencil, equivalent to a simple gradient operator. Uses
    GPU acceleration when available.

    Parameters
    ----------
    array : ArrayLike
        Input array to filter (2D to 5D supported). Will be cast to float32.
    axis : int
        Axis along which to compute the edge filter.
    gpu : bool
        If True and a CUDA GPU is available, use GPU acceleration.

    Returns
    -------
    numpy.ndarray
        Filtered array of the same shape as input, containing the
        difference between forward and backward neighbors along the axis.
    """
    # Cast to float:
    array = array.astype(dtype=numpy.float32, copy=False)

    # Get the appropriate pre-defined stencil kernel
    ndim = array.ndim
    kernel_key = (ndim, axis)

    if kernel_key not in _STENCIL_KERNELS:
        raise ValueError(
            f"Unsupported combination: {ndim}D array with axis={axis}. "
            f"Supported: 2D-5D arrays with axis in range [0, ndim)."
        )

    _kernel = _STENCIL_KERNELS[kernel_key]

    def _fast_edge_filter(array: ArrayLike):
        array = _kernel(array)
        return array

    if gpu and numba.cuda.is_available():
        from numba import njit

        edge_func = njit(parallel=False)(_fast_edge_filter)
    else:
        edge_func = jit(parallel=True)(_fast_edge_filter)

    return edge_func(array)
