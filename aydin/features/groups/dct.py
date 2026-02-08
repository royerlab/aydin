"""Discrete Cosine Transform (DCT) feature group."""

import numpy
from numpy.linalg import norm
from scipy.fft import idstn

from aydin.features.groups.correlation import CorrelationFeatures


class DCTFeatures(CorrelationFeatures):
    """
    DCT Feature Group class

    Generates Discrete Cosine Transform (DCT) features.
    """

    def __init__(self, size: int, max_freq: float = 0.75, power: float = 0.5):
        """
        Constructor that configures these features.

        Parameters
        ----------
        size : int
            Size of the DCT filters
        max_freq : float
            Maximum frequency of DCT filters
            (advanced)
        power : float
            Filters can be exponentiated to a given power to change behaviour.
            (advanced)

        """
        super().__init__(kernels=None)
        self.size = size
        self.max_freq = max_freq
        self.power = power

        self.image = None
        self.exclude_center: bool = False

    def _ensure_dct_kernels_available(self, ndim: int):
        """Ensure DCT kernels are computed for the given dimensionality.

        Constructs inverse DCT basis functions as correlation kernels,
        filtered by ``max_freq`` and power-transformed by ``power``.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        """
        if self.kernels is None or self.kernels[0].ndim != ndim:
            dct_kernels = []
            shape = tuple((self.size,) * ndim)
            dctcoefs = numpy.zeros(shape=shape)

            for index, _ in numpy.ndenumerate(dctcoefs):

                freqs = numpy.array([i / self.size for i in index])
                freq = norm(freqs)

                if freq > self.max_freq:
                    continue

                dctcoefs[...] = 0
                dctcoefs[index] = 1

                kernel = idstn(dctcoefs, norm="ortho")
                kernel /= kernel.std()
                kernel = numpy.sign(kernel) * abs(kernel) ** self.power
                kernel /= kernel.std()
                kernel += kernel.min()
                kernel /= kernel.sum()
                kernel = kernel.astype(numpy.float32)

                dct_kernels.append(kernel)

            self.kernels = dct_kernels

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius based on the DCT filter size.

        Returns
        -------
        radius : int
            Half the filter size.
        """
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        """Return the number of DCT features for the given dimensionality.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.

        Returns
        -------
        num : int
            Number of DCT features (depends on ``max_freq`` and ``size``).
        """
        self._ensure_dct_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare DCT features by constructing kernels for the image.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which features will be computed.
        excluded_voxels : list of tuple of int, optional
            Voxels to exclude from feature computation.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        if excluded_voxels is None:
            excluded_voxels = []

        self._ensure_dct_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
