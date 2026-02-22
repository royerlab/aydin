"""Abstract base class for feature groups used by extensible feature generators."""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class FeatureGroupBase(ABC):
    """Abstract base class for feature groups.

    A feature group encapsulates a family of related features that share the
    same computation strategy (e.g., uniform filtering, median filtering,
    convolution with learned kernels). Feature groups are added to an
    ``ExtensibleFeatureGenerator`` to compose the full feature set.

    Subclasses must implement ``receptive_field_radius``, ``num_features``,
    ``prepare``, and ``compute_feature``.
    """

    def __init__(self):
        """Construct a feature group."""

    @property
    @abstractmethod
    def receptive_field_radius(self) -> int:
        """Return the receptive field radius in pixels.

        Returns
        -------
        radius : int
            Maximum distance (in pixels) from the center voxel that
            features in this group can reach.
        """
        raise NotImplementedError()

    @abstractmethod
    def num_features(self, ndim: int) -> int:
        """Return the number of features produced for the given dimensionality.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the image.

        Returns
        -------
        num : int
            Number of features this group will produce.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare(
        self, image, excluded_voxels: Sequence[Tuple[int, ...]] = None, **kwargs
    ):
        """Prepare the feature group for computation on the given image.

        Pre-computes any shared state (e.g., filtered images, kernels) that
        will be reused across individual feature computations.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which features will be computed.
        excluded_voxels : Sequence[Tuple[int, ...]], optional
            Voxels to exclude from features, as a list of coordinate tuples
            relative to the center voxel.
        **kwargs
            Additional keyword arguments for feature computation.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_feature(self, index: int, feature):
        """Compute a single feature by index, storing the result in-place.

        The feature index must be strictly less than the number of features
        returned by ``num_features``.

        Parameters
        ----------
        index : int
            Index of the feature to compute within this group.
        feature : numpy.ndarray
            Pre-allocated array into which the computed feature is stored.
        """
        raise NotImplementedError()

    def finish(self):
        """Clean up resources allocated during feature computation.

        After cleanup, this feature group can be reused to compute features
        for a new image. The default implementation is a no-op.
        """
        # By default there is nothing to free:
        pass
