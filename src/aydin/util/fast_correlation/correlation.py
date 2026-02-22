"""Fast n-dimensional correlation with Numba CPU fallback to SciPy."""

import traceback

from numpy.typing import ArrayLike
from scipy.ndimage import correlate as scipy_ndimage_correlate

from aydin.util.fast_correlation.numba_cpu import numba_cpu_correlate


def correlate(image: ArrayLike, weights: ArrayLike, output: ArrayLike = None):
    """Correlate an image with a kernel using the fastest available method.

    Attempts to use the Numba-accelerated CPU implementation first. If that
    fails (e.g., unsupported dimensionality), falls back to SciPy's
    ``ndimage.correlate``.

    Parameters
    ----------
    image : ArrayLike
        Input image array.
    weights : ArrayLike
        Correlation kernel. Must have the same number of dimensions as ``image``
        and odd lengths along all axes for the Numba implementation.
    output : ArrayLike, optional
        Pre-allocated output array. If None, a new array is created.

    Returns
    -------
    numpy.ndarray
        Correlated image of the same shape as ``image``.
    """

    try:
        output = numba_cpu_correlate(image=image, kernel=weights, output=output)
        return output
    except Exception:
        traceback.print_exc()
        return scipy_ndimage_correlate(image=image, kernel=weights, output=output)
