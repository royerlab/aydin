from aydin.it.normalisers.base import NormaliserBase


class IdentityNormaliser(NormaliserBase):
    """Identity Normaliser"""

    def __init__(self, **kwargs):
        """Constructs a normalisers"""
        super().__init__(**kwargs)

    def calibrate(self, array):
        """Calibrate method

        Parameters
        ----------
        array : numpy.ndarray

        """
        self.original_dtype = array.dtype

        return None, None
