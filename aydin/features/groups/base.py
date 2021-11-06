from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class FeatureGroupBase(ABC):
    """
    Feature Group base class
    """

    def __init__(self):
        """
        Constructs a feature group
        """

    @property
    @abstractmethod
    def receptive_field_radius(self) -> int:
        """
        Returns the receptive field radius in voxels

        Parameters
        ----------

        Returns
        -------
        result : int
            receptive field radius in pixels

        """
        raise NotImplementedError()

    @abstractmethod
    def num_features(self, ndim: int) -> int:
        """
        Returns the number of features given an image dimension

        Parameters
        ----------
        ndim : Number of dimension sof image for which features will be computed

        Returns
        -------
        result : int
            Number of features

        """
        raise NotImplementedError()

    @abstractmethod
    def prepare(
        self, image, excluded_voxels: Sequence[Tuple[int, ...]] = None, **kwargs
    ):
        """
        Prepares the computation of the features in the group.
        Sets the image for which features should be computed.

        Parameters
        ----------
        image : Image for which features will be computed
        excluded_voxels : voxels to exclude from feature as list of coordinate tuples.
        kwargs : key-value arguments for feature functions

        Returns
        -------

        """
        raise NotImplementedError()

    @abstractmethod
    def compute_feature(self, index: int, feature):
        """
        Computes feature of a given index. The feature index must be strictly less than the number of features returned
         by get_num_features.

        Parameters
        ----------
        index : index for feature
        feature : array into which to store feature

        Returns
        -------


        """
        raise NotImplementedError()

    def finish(self):
        """
        Cleans up and frees any resource allocated during feature computation. After cleanup this feature group can be reused to compute features for a new image.

        Parameters
        ----------

        Returns
        -------


        """
        # By default there is nothing to free:
        pass
