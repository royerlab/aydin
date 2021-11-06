from typing import Sequence, Tuple

import numpy
from scipy.ndimage import median_filter

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint


class MedianFeatures(FeatureGroupBase):
    """
    Median Feature Group class
    """

    def __init__(self, radii: Sequence[int]):
        super().__init__()
        self.radii = tuple(radii)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []

        self.kwargs = None

    @property
    def receptive_field_radius(self) -> int:
        radius = max(r for r in self.radii)
        return radius

    def num_features(self, ndim: int) -> int:
        return len(self.radii)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image
        self.excluded_voxels = excluded_voxels
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        radius = self.radii[index]
        lprint(
            f"Median feature: {index}, radius: {radius}, excluded_voxels={self.excluded_voxels}"
        )
        footprint_shape = (2 * radius + 1,) * self.image.ndim
        footprint = numpy.ones(footprint_shape)

        # We need to modify the footprint to take into account the excluded voxels:
        for excluded_voxel in self.excluded_voxels:

            # First we check if the excluded voxel falls within the footprint of the feature:
            if all(
                (
                    -s // 2 <= v <= s // 2
                    for s, v in zip(footprint.shape, excluded_voxel)
                )
            ):
                # Here is the coordinate of the excluded voxel
                coord = tuple(
                    (s // 2 + v for s, v in zip(footprint.shape, excluded_voxel))
                )

                aslice = tuple((slice(i, i + 1, None) for i in coord))
                footprint[aslice] = 0

        median_filter(self.image, footprint=footprint, output=feature)

    def finish(self):
        # Here we cleanup any resource alocated for the last feature computation:
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
