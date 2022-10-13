from typing import Sequence, Optional, Tuple, Callable

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.fast_correlation.correlation import correlate as fast_correlate
from aydin.util.log.log import lprint


class CorrelationFeatures(FeatureGroupBase):
    """
    Correlation (convolutional) Feature Group class

    Generates correlative features given a set of kernels.
    """

    def __init__(
        self,
        kernels: Optional[Sequence[ArrayLike]],
        separable: bool = True,
        _correlate_func: Callable = None,
    ):
        """
        Constructor that configures these features.

        Parameters
        ----------
        kernels : Optional[Sequence[ArrayLike]]
            Sequence of kernels to use to compute features.

        separable : bool
            If True the kernels are assumed to be separable as identical 1d
            kernels for each axis.

        """
        super().__init__()
        self.kernels = kernels if kernels is None else list(kernels)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []
        self.separable = separable
        self.kwargs = None

        if _correlate_func is None:
            _correlate_func = fast_correlate
            # _correlate_func = correlate

        self._correlate_func = _correlate_func

    @property
    def receptive_field_radius(self) -> int:
        radius = max(max(s // 2 for s in k.shape) for k in self.kernels)
        return radius

    def num_features(self, ndim: int) -> int:
        return len(self.kernels)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image
        self.excluded_voxels = list(excluded_voxels)
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        kernel = self.kernels[index]
        lprint(
            f"Correlative feature: {index} of shape={kernel.shape}, excluded_voxels={self.excluded_voxels}"
        )

        if len(self.excluded_voxels) > 0:
            # No need to do crazy stuff if there is no excluded voxels to start with...

            # We keep the weights that we zero out:
            center_weights = []

            # We keep the slice object for accessing the excluded voxels in the kernel:
            slices = []

            # We need to modify the kernel to take into account the excluded voxels:
            for excluded_voxel in self.excluded_voxels:

                # First we check if the excluded voxel falls within the footprint of the feature:
                if all(
                    (
                        -s // 2 <= v <= s // 2
                        for s, v in zip(kernel.shape, excluded_voxel)
                    )
                ):
                    # Here is the coordinate of the excluded voxel
                    coord = tuple(
                        (s // 2 + v for s, v in zip(kernel.shape, excluded_voxel))
                    )

                    # slice to address the kernel array:
                    aslice = tuple((slice(c, c + 1, None) for c in coord))
                    slices.append(aslice)
                    center_weights.append(float(kernel[aslice]))
                    kernel[aslice] = 0

            # We use a holed-gaussian estimate of the missing value,
            # so we do use the correct weight but avoid using the actual center value
            missing = numpy.zeros_like(kernel)
            for aslice, weight in zip(slices, center_weights):
                missing[aslice] = weight

            # We apply a Gaussian filter to find neighbooring voxels from which we can estimate the missing values:
            missing = gaussian_filter(missing, sigma=0.5)

            # Save the sum so:
            saved_sum = missing.sum()

            # We zero the excluded voxels from it:
            for aslice in slices:
                missing[aslice] = 0

            # We rescale the missing value estimation kernel:
            missing *= saved_sum / missing.sum()

            # We add the missing-value-estimation to the kernel:
            kernel += missing

        # Convolution:
        self._correlate(
            image=self.image, kernel=kernel, separable=self.separable, output=feature
        )

        # if self.image.size > 4**4:
        #     import napari
        #     from napari import Viewer
        #     viewer = Viewer()
        #     viewer.add_image(self.image, name='self.image')
        #     viewer.add_image(feature, name='feature')
        #     napari.run()

    def finish(self):
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
        self.kernels = None

    def _correlate(
        self, image: ArrayLike, kernel: ArrayLike, separable: bool, output: ArrayLike
    ):

        if separable and kernel.ndim == 1:

            # Looping through the axis:
            for axis in range(image.ndim):

                # prepare shape:
                shape = [1] * image.ndim
                shape[axis] = kernel.shape[0]

                # reshape kernel:
                kernel = kernel.reshape(*shape)

                # convolve:
                correlate_output = output if axis == image.ndim - 1 else None
                image = self._correlate_func(
                    image, weights=kernel, output=correlate_output
                )

        else:
            self._correlate_func(image, weights=kernel, output=output)
