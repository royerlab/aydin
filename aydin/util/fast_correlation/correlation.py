import traceback

from numpy.typing import ArrayLike
from scipy.ndimage import correlate as scipy_ndimage_correlate

from aydin.util.fast_correlation.numba_cpu import numba_cpu_correlate


def correlate(image: ArrayLike, weights: ArrayLike, output: ArrayLike = None):

    try:
        output = numba_cpu_correlate(image=image, kernel=weights, output=output)
        return output
    except Exception:
        traceback.print_exc()
        return scipy_ndimage_correlate(image=image, kernel=weights, output=output)
