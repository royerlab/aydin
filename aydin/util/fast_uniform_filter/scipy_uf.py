"""SciPy-based uniform filter wrapper with dtype handling."""

import numpy
from scipy.ndimage import uniform_filter


def scipy_uniform_filter(
    image, size=3, output=None, mode="nearest", cval=0.0, origin=0
):
    """Apply a uniform (box) filter using SciPy with dtype preservation.

    Wraps ``scipy.ndimage.uniform_filter`` with automatic handling of
    float16 arrays (which SciPy does not support natively).

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    size : int or tuple of int
        Filter window size.
    output : numpy.ndarray, optional
        Pre-allocated output array (unused, kept for API compatibility).
    mode : str
        Boundary mode for the filter. Default is 'nearest'.
    cval : float
        Constant value for 'constant' boundary mode.
    origin : int
        Filter origin offset.

    Returns
    -------
    numpy.ndarray
        Filtered image with the original dtype.
    """
    # Save original image dtype:
    original_dtype = image.dtype

    # Scipy does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype
    image = image.astype(dtype=dtype, copy=False)

    output = uniform_filter(image, size=size, mode=mode, cval=cval, origin=origin)

    output = output.astype(dtype=original_dtype, copy=False)

    return output
