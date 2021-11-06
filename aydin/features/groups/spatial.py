from math import sin, pi
from typing import Optional, Tuple

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint


class SpatialFeatures(FeatureGroupBase):
    def __init__(self, coarsening: int = 1, period: float = 0):
        super().__init__()

        self.coarsening = coarsening
        self.period = period

        self.image = None
        self.offset: Optional[Tuple[float, ...]] = None
        self.scale: Optional[Tuple[float, ...]] = None

    @property
    def receptive_field_radius(self) -> int:
        return 0

    def num_features(self, ndim: int) -> int:
        return ndim

    def prepare(
        self,
        image,
        offset: Optional[Tuple[float, ...]] = None,
        scale: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        self.image = image
        self.offset = offset
        self.scale = scale
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        lprint(
            f"Spatial feature {index}, offset={self.offset}, scale={self.scale}, coarsening={self.coarsening}, period={self.period} "
        )

        # Axis and dimensions:
        axis: int = index
        ndim = self.image.ndim

        # Scale and offset:
        offset = (0,) * ndim if self.offset is None else self.offset
        scale = (1,) * ndim if self.scale is None else self.scale

        # This is the offset for the corresponding axis:
        offset = offset[axis]
        scale = scale[axis]

        # Get the length on the dimension we want to iterate over
        dim_length = self.image.shape[axis]

        # Cache period and coarsening in local variables:
        coarsening = self.coarsening
        period = self.period

        # Iterate over the dimension of interest
        for index in range(dim_length):
            # Prepare a placeholder 'ndim' slice object
            slicing = [slice(None)] * len(feature.shape)

            # Change the slicing on the dimension we are iterating over
            slicing[axis] = slice(index, index + 1, 1)

            # Compute value and adds offset,
            # and we coarsen the resolution by a factor of two so that individual pixels cannot be identified:
            value = scale * float(coarsening * ((offset + index) // coarsening))

            if period != 0:
                value = sin((2 * pi * value) / period)

            # Put the normalized position value across dimensions we are NOT iterating over
            feature[tuple(slicing)] = value

    def finish(self):
        # Here we cleanup any resource allocated for the last feature computation:
        self.image = None
