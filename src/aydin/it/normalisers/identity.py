"""Identity normaliser that performs no value transformation.

Provides `IdentityNormaliser`, a no-op normaliser that preserves the
original data values while still tracking the original dtype.
"""

from aydin.it.normalisers.base import NormaliserBase


class IdentityNormaliser(NormaliserBase):
    """Identity normaliser that performs no value transformation.

    Preserves original data values unchanged. Useful when normalization
    is not desired but the normaliser interface is still required.
    """

    def __init__(self, **kwargs):
        """Construct an IdentityNormaliser.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to NormaliserBase.
        """
        super().__init__(**kwargs)

    def calibrate(self, array):
        """Calibrate by recording the original dtype without computing a range.

        Parameters
        ----------
        array : numpy.ndarray
            Array to record the dtype from.

        Returns
        -------
        tuple of (None, None)
            No normalization range is computed.
        """
        self.original_dtype = array.dtype

        return None, None
