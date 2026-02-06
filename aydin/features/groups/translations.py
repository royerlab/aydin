"""Translation feature group that generates shifted copies of the image."""

from typing import Sequence, Tuple

from scipy.ndimage import shift

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint


class TranslationFeatures(FeatureGroupBase):
    """
    Translations Feature Group class

    These features are just the image itself translated by a set of vectors.
    """

    def __init__(self, translations: Sequence[Tuple[int, ...]]):
        """
        Constructor that configures these features.

        Parameters
        ----------
        translations : Sequence[Tuple[int, ...]]
            Sequence of translation vectors.

        """
        super().__init__()
        self.translations = list(translations)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []

        self.kwargs = None

    @property
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius based on the largest translation.

        Returns
        -------
        radius : int
            Maximum absolute translation component across all vectors.
        """
        radius = max(max(abs(d) for d in t) for t in self.translations)
        return radius

    def num_features(self, ndim: int) -> int:
        """Return the number of translation features (one per translation vector).

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions (unused, included for API consistency).

        Returns
        -------
        num : int
            Number of translation vectors.
        """
        return len(self.translations)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare the translation feature group for computation.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which features will be computed.
        excluded_voxels : list of tuple of int, optional
            Voxels to exclude. If a translation matches an excluded voxel,
            that feature produces no output.
        **kwargs
            Additional keyword arguments (unused).
        """
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image
        self.excluded_voxels = excluded_voxels
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
        """Compute a translated copy of the image as a feature.

        If the translation matches any excluded voxel, the feature is
        skipped (the output array is left unchanged).

        Parameters
        ----------
        index : int
            Index of the translation vector to apply.
        feature : numpy.ndarray
            Pre-allocated array for storing the translated image.
        """
        translation = self.translations[index]
        lprint(
            f"translation feature: {index}, translation={translation}, exclude_center={self.excluded_voxels}"
        )

        for excluded_voxel in self.excluded_voxels:
            if all(ev == t for ev, t in zip(excluded_voxel, translation)):
                return

        shift(
            self.image,
            shift=list(translation),
            output=feature,
            order=0,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )

    def finish(self):
        """Clean up references from the last feature computation."""
        # Here we cleanup any resource allocated for the last feature computation:
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
