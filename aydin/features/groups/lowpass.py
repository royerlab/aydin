"""Low-pass Butterworth filter feature group."""

import numpy

from aydin.features.groups.correlation import CorrelationFeatures
from aydin.it.classic_denoisers.butterworth import denoise_butterworth


class LowPassFeatures(CorrelationFeatures):
    """Low-pass Butterworth filter feature group.

    Generates features by convolving the image with Butterworth low-pass
    impulse-response kernels at regularly spaced frequency cutoffs. Each
    kernel captures progressively lower frequency information, making this
    feature group highly effective for images where Butterworth denoising
    already works well.

    Attributes
    ----------
    sizes : list of int
        Spatial sizes (odd integers) for each Butterworth kernel.
    freq_cutoffs : list of float
        Frequency cutoff for each kernel, linearly spaced between
        ``max_freq`` and ``min_freq``.
    order : float
        Butterworth filter order controlling roll-off steepness.
    exclude_center : bool
        Whether the center pixel is excluded from features.
    """

    def __init__(
        self,
        num_features: int = 9,
        min_size: int = 5,
        max_size: int = 9,
        min_freq: float = 0.50,
        max_freq: float = 0.90,
        order: float = 2,
        separable: bool = False,
    ):
        """
        Constructor that configures these features.

        Parameters
        ----------

        num_features : int
            Number of features.

        min_size : int
            Minimum size of the low-pass filters.

        max_size : int
            Maximum size of the low-pass filters.

        min_freq : float
            Minimum cut-off frequency.
            (advanced)

        max_freq : float
            Maximum cut-off frequency.
            (advanced)

        order : float
            Butterworth filter order.
            (advanced)

        separable : bool
            If True the kernels are assumed to be separable as identical 1d
            kernels for each axis.

        """
        super().__init__(kernels=None, separable=separable)

        self.min_size = min_size
        self.max_size = max_size
        self._num_features = num_features
        if num_features == 1:
            self.sizes = [(int(round(min_size)) // 2) * 2 + 1]
            self.freq_cutoffs = [max_freq]
        else:
            self.sizes = [
                (
                    int(
                        round(
                            min_size + ((max_size - min_size) / (num_features - 1)) * i
                        )
                    )
                    // 2
                )
                * 2
                + 1
                for i in range(num_features)
            ]
            self.freq_cutoffs = [
                max_freq - ((max_freq - min_freq) / (num_features - 1)) * i
                for i in range(num_features)
            ]
        self.order = order

        self.image = None
        self.exclude_center: bool = False

    def _ensure_random_kernels_available(self, ndim: int):
        """Ensure Butterworth low-pass kernels are computed
        for the given dimensionality.

        Constructs the kernels lazily on first call or when the dimensionality
        changes. Each kernel is a Butterworth impulse response at a different
        frequency cutoff.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        """
        # if we are in the 'separable' case, we only need to generate 1d kernels:
        ndim = 1 if self.separable else ndim

        if self.kernels is None or self.kernels[0].ndim != ndim:
            lowpass_kernels = []

            num_features = self._num_features

            for index in range(num_features):
                size = self.sizes[index]
                shape = tuple((size,) * ndim)
                freq_cutoff = self.freq_cutoffs[index]
                kernel = numpy.zeros(shape=shape, dtype=numpy.float32)
                kernel[(size // 2,) * ndim] = 1.0
                kernel = denoise_butterworth(
                    kernel, freq_cutoff=freq_cutoff, order=self.order
                )
                kernel /= kernel.sum()

                # import napari
                # from napari import Viewer
                # viewer = Viewer()
                # viewer.add_image(kernel, name='kernel')
                # napari.run()

                lowpass_kernels.append(kernel)

            self.kernels = lowpass_kernels

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius based on the largest filter size.

        Returns
        -------
        radius : int
            Half the maximum filter size.
        """
        return max(self.sizes) // 2

    def num_features(self, ndim: int) -> int:
        """Return the number of low-pass features.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.

        Returns
        -------
        num : int
            Number of low-pass features.
        """
        self._ensure_random_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare low-pass features by constructing kernels for the image.

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
        self._ensure_random_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
