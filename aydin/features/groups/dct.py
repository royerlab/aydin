import numpy
from numpy.linalg import norm
from scipy.fft import idstn

from aydin.features.groups.convolutional import ConvolutionalFeatures


class DCTFeatures(ConvolutionalFeatures):
    """
    DCT Feature Group class
    """

    def __init__(self, size: int, max_freq: float = 0.75, power: float = 0.5):
        super().__init__(kernels=None)
        self.size = size
        self.max_freq = max_freq
        self.power = power

        self.image = None
        self.exclude_center: bool = False

    def _ensure_dct_kernels_available(self, ndim: int):
        # Ensures that the kernels are available for subsequent steps.
        # We can't construct the kernels until we know the dimension of the image
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
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        self._ensure_dct_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self._ensure_dct_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
