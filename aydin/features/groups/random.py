"""Random convolutional feature group for experimental feature generation."""

from typing import Optional, Sequence, Tuple

import numpy
from numpy.random import rand
from scipy.ndimage import gaussian_filter

from aydin.features.groups.correlation import CorrelationFeatures


class RandomFeatures(CorrelationFeatures):
    """
    Random Feature Group class

    Generates features by convolving the image with a random set of filters.
    """

    def __init__(self, size: int, num_features: Optional[int] = None):
        """
        Constructor that configures these features.

        Parameters
        ----------

        size : int
            Size of the kernels used.

        num_features : Optional[int]
            Number of features to generate. If None the maximum number of features
            that may be linearly independent is used.
        """
        super().__init__(kernels=None)
        self.size = size
        self._num_features = num_features

        self.image = None
        self.exclude_center: Sequence[Tuple[int, ...]] = []

    def _ensure_random_kernels_available(self, ndim: int):
        """Ensure random kernels are generated for the given dimensionality.

        Generates deterministic pseudo-random kernels (seeded by kernel index)
        consisting of binary patterns smoothed by a Gaussian filter.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        """
        if self.kernels is None or self.kernels[0].ndim != ndim:
            rnd_kernels = []

            num_features = (
                self.size**ndim if self._num_features is None else self._num_features
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

                rnd_kernels.append(kernel)

            self.kernels = rnd_kernels

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius based on the kernel size.

        Returns
        -------
        radius : int
            Half the kernel size.
        """
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        """Return the number of random convolutional features.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.

        Returns
        -------
        num : int
            Number of random features.
        """
        self._ensure_random_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare random features by generating kernels for the image.

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
