"""Percentile normaliser for robust value normalization.

Provides `PercentileNormaliser` which normalizes image values to [0, 1]
using percentile-based range estimation, making it robust to outliers.
Supports both NumPy and Dask arrays.
"""

import math

import dask
import numpy

from aydin.it.normalisers.base import NormaliserBase
from aydin.util.log.log import lprint, lsection


class PercentileNormaliser(NormaliserBase):
    """Percentile-based normaliser for robust value normalization.

    Uses percentile values instead of absolute min/max to determine
    the normalization range, making it robust to outlier values.
    If no percentile is specified, an optimal value is computed
    based on the array size.
    """

    percent: float

    def __init__(self, percentile: float = None, **kwargs):
        """Construct a PercentileNormaliser.

        Parameters
        ----------
        percentile : float, optional
            Percentile value for range estimation (e.g. 0.01 for 1%).
            If None, an optimal percentile is computed based on array size.
        **kwargs
            Additional keyword arguments passed to NormaliserBase.
        """
        super().__init__(**kwargs)

        self.percentile = percentile

    def calibrate(self, array):
        """Calibrate by computing percentile-based min and max of the array.

        Supports both NumPy and Dask arrays.

        Parameters
        ----------
        array : numpy.ndarray or dask.array.Array
            Array to compute the normalization range from.

        Returns
        -------
        tuple of (float, float)
            The computed (min_value, max_value) range based on percentiles.
        """

        with lsection("Calibrating array using percentile method"):
            self.original_dtype = array.dtype

            if self.percentile is None:
                # We compute an ideal percentile for this array given the size:
                size = array.size
                p = min(0.001, 0.01 * math.sqrt(size) / size)
            else:
                p = self.percentile

            lprint(f"Using percentile value: {p}")

            if hasattr(array, '__dask_keys__'):
                self.rmin = dask.array.percentile(array.flatten(), 100 * p).compute()
                self.rmax = dask.array.percentile(
                    array.flatten(), 100 - 100 * p
                ).compute()
            else:
                self.rmin = numpy.percentile(array, 100 * p)
                self.rmax = numpy.percentile(array, 100 - 100 * p)

            lprint(f"Range for normalisation: [{self.rmin}, {self.rmax}]")

            return self.rmin, self.rmax
