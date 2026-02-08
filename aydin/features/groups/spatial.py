"""Spatial coordinate feature group for position-dependent denoising."""

from math import pi, sin
from typing import Optional, Tuple

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import aprint


class SpatialFeatures(FeatureGroupBase):
    """
    Spatial Feature Group class

    These features are simply the shifted, scaled, and possibly quantised
    coordinates of the voxels themselves.
    """

    def __init__(self, coarsening: int = 1, period: float = 0):
        """
        Constructor that configures these features.

        Parameters
        ----------
        coarsening : int
            How many pixels should have the same value?
            (advanced)
        period : float
            If not zero then the feature is sinusoidal.
            (advanced)
        """
        super().__init__()

        self.coarsening = coarsening
        self.period = period

        self.image = None
        self.offset: Optional[Tuple[float, ...]] = None
        self.scale: Optional[Tuple[float, ...]] = None

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius (always 0 for spatial features).

        Returns
        -------
        radius : int
            Always 0, since spatial features do not depend on neighboring pixels.
        """
        return 0

    def num_features(self, ndim: int) -> int:
        """Return the number of spatial features (one per spatial dimension).

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.

        Returns
        -------
        num : int
            Equal to ``ndim``.
        """
        return ndim

    def prepare(
        self,
        image,
        offset: Optional[Tuple[float, ...]] = None,
        scale: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        """Prepare spatial feature computation for the given image.

        Parameters
        ----------
        image : numpy.ndarray
            Image whose shape determines the spatial feature dimensions.
        offset : tuple of float, optional
            Offset to add to each spatial coordinate (one per dimension).
        scale : tuple of float, optional
            Scale factor to multiply each spatial coordinate (one per dimension).
        **kwargs
            Additional keyword arguments (unused).
        """
        self.image = image
        self.offset = offset
        self.scale = scale
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        """Compute a spatial coordinate feature for the given axis.

        Each feature contains the (optionally shifted, scaled, and quantised)
        coordinate values along one spatial dimension. When ``period > 0``,
        the coordinates are passed through a sinusoidal function.

        Parameters
        ----------
        index : int
            Index of the spatial dimension to generate a feature for.
        feature : numpy.ndarray
            Pre-allocated array for storing the computed feature.
        """
        aprint(
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
        """Clean up references from the last feature computation."""
        # Here we cleanup any resource allocated for the last feature computation:
        self.image = None
