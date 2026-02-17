"""Median filter feature group for robust denoising features."""

from typing import Sequence, Tuple

import numpy
from scipy.ndimage import median_filter

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import aprint


class MedianFeatures(FeatureGroupBase):
    """Median filter feature group.

    Generates features by applying median filters of increasing sizes
    (radii). Median filters are robust to outliers and salt-and-pepper
    noise, making these features useful for recovering smooth structures
    in heavily corrupted images.

    Attributes
    ----------
    radii : tuple of int
        Radii of the median filters. Each radius produces one feature
        with a ``(2*radius+1)``-sized hypercubic footprint.
    image : numpy.ndarray or None
        Reference to the current image being processed.
    excluded_voxels : list of tuple of int
        Voxels excluded from filter footprints for blind-spot denoising.
    """

    def __init__(self, radii: Sequence[int]):
        """
        Constructor that configures these features.

        Parameters
        ----------
        radii : Sequence[int]
            Sequence of radii to be used for the median filters.
        """
        super().__init__()
        self.radii = tuple(radii)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []

        self.kwargs = None

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius (largest median filter radius).

        Returns
        -------
        radius : int
            Maximum radius among all configured median filters.
        """
        radius = max(r for r in self.radii)
        return radius

    def num_features(self, ndim: int) -> int:
        """Return the number of median features (one per radius).

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions (unused, included for API consistency).

        Returns
        -------
        num : int
            Number of median filter radii configured.
        """
        return len(self.radii)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare the median feature group for computation.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which median features will be computed.
        excluded_voxels : list of tuple of int, optional
            Voxels to exclude from the median filter footprints.
        **kwargs
            Additional keyword arguments (unused).
        """
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image
        self.excluded_voxels = excluded_voxels
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        """Compute a median-filtered feature for the given radius index.

        Applies a median filter with the configured radius. Excluded voxels
        are zeroed out in the filter footprint before applying.

        Parameters
        ----------
        index : int
            Index into the radii list selecting which median filter to apply.
        feature : numpy.ndarray
            Pre-allocated array for storing the median-filtered result.
        """
        radius = self.radii[index]
        aprint(
            f"Median feature: {index}, radius: {radius}, "
            f"excluded_voxels={self.excluded_voxels}"
        )
        footprint_shape = (2 * radius + 1,) * self.image.ndim
        footprint = numpy.ones(footprint_shape)

        # We need to modify the footprint to take into account the excluded voxels:
        for excluded_voxel in self.excluded_voxels:

            # First we check if the excluded voxel falls
            # within the footprint of the feature:
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
        """Clean up references from the last feature computation."""
        # Here we cleanup any resource allocated for the last feature computation:
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
