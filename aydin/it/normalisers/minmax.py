"""Min-max normaliser for value normalization.

Provides `MinMaxNormaliser` which normalizes image values to [0, 1]
using the full min and max of the data. Supports both NumPy and Dask arrays.
"""

import dask
import numpy

from aydin.it.normalisers.base import NormaliserBase
from aydin.util.log.log import aprint, asection


class MinMaxNormaliser(NormaliserBase):
    """Min-max normaliser that maps the full data range to [0, 1].

    Uses the absolute minimum and maximum values of the calibration
    array to define the normalization range.
    """

    def __init__(self, **kwargs):
        """Construct a MinMaxNormaliser.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to NormaliserBase.
        """
        super().__init__(**kwargs)

    def calibrate(self, array):
        """Calibrate by computing the min and max of the array.

        Supports both NumPy and Dask arrays.

        Parameters
        ----------
        array : numpy.ndarray or dask.array.Array
            Array to compute the normalization range from.

        Returns
        -------
        tuple of (float, float)
            The computed (min_value, max_value) range.
        """
        with asection("Calibrating array using minmax method"):
            self.original_dtype = array.dtype

            if hasattr(array, '__dask_keys__'):
                self.rmin = dask.array.min(array.flatten()).compute()
                self.rmax = dask.array.max(array.flatten()).compute()
            else:
                self.rmin = numpy.min(array)
                self.rmax = numpy.max(array)

            aprint(f"Range for normalisation: [{self.rmin}, {self.rmax}]")

            return self.rmin, self.rmax
