import numpy


from aydin.features.groups.correlation import CorrelationFeatures
from aydin.it.classic_denoisers.butterworth import denoise_butterworth


class LowPassFeatures(CorrelationFeatures):
    """
    Low-Pass Feature Group class

    Generates Low-Pass features using Butterworth filtering.
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

        min_size: int
            Minimum size of the low-pass filters.

        max_size: int
            Maximu size of the low-pass filters.

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
        self.sizes = [
            (
                int(round(min_size + ((max_size - min_size) / (num_features - 1)) * i))
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
        # Ensures that the kernels are available for subsequent steps.
        # We can't construct the kernels until we know the dimension of the image.

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
        return max(self.sizes) // 2

    def num_features(self, ndim: int) -> int:
        self._ensure_random_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []
        self._ensure_random_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
