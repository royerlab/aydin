from typing import Sequence, Optional, Tuple

import numpy
from numpy import ndarray
from scipy.ndimage import convolve, gaussian_filter

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint


class ConvolutionalFeatures(FeatureGroupBase):
    """
    Convolutional Feature Group class
    """

    def __init__(self, kernels: Optional[Sequence[ndarray]]):
        super().__init__()
        self.kernels = kernels if kernels is None else list(kernels)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []

        self.kwargs = None

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
            f"Convolutional feature: {index} of shape={kernel.shape}, excluded_voxels={self.excluded_voxels}"
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
            missing = gaussian_filter(missing, sigma=1)

            # We zero the excluded voxels from it:
            for aslice in slices:
                missing[aslice] = 0

            # We rescale the missing value estimation kernel:
            missing /= missing.sum()

            # We add the missing-value-estimation to the kernel:
            kernel += missing

        # Convolution:
        convolve(self.image, weights=kernel, output=feature)

    def finish(self):
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
        self.kernels = None
