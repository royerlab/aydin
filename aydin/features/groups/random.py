from typing import Optional, Sequence, Tuple

import numpy
from numpy.random import rand
from scipy.ndimage import gaussian_filter

from aydin.features.groups.convolutional import ConvolutionalFeatures


class RandomFeatures(ConvolutionalFeatures):
    """
    Random Feature Group class
    """

    def __init__(self, size: int, num_features: Optional[int] = None):
        super().__init__(kernels=None)
        self.size = size
        self._num_features = num_features

        self.image = None
        self.exclude_center: Sequence[Tuple[int, ...]] = []

    def _ensure_random_kernels_available(self, ndim: int):
        # Ensures that the kernels are available for subsequent steps.
        # We can't construct the kernels until we know the dimension of the image
        if self.kernels is None or self.kernels[0].ndim != ndim:
            dct_kernels = []

            num_features = (
                self.size ** ndim if self._num_features is None else self._num_features
            )

            shape = tuple((self.size,) * ndim)

            for k in range(num_features):
                numpy.random.seed(k)
                kernel = numpy.where(rand(*shape) > 0.5, 1.0, 0.0)
                kernel = gaussian_filter(kernel, sigma=0.5)
                kernel /= kernel.sum()
                kernel = kernel.astype(numpy.float32, copy=False)

                # import napari
                # with napari.gui_qt():
                #      from napari import Viewer
                #      viewer = Viewer()
                #      viewer.add_image(kernel, name='kernel')

                dct_kernels.append(kernel)

            self.kernels = dct_kernels

    @property
    def receptive_field_radius(self) -> int:
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        self._ensure_random_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []
        self._ensure_random_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
