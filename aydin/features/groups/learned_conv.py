"""Learned convolutional feature group using MiniBatchKMeans kernel extraction."""

from typing import Optional, Union

from aydin.features.groups.correlation import CorrelationFeatures
from aydin.features.groups.extract_kernels import extract_kernels


class LearnedCorrelationFeatures(CorrelationFeatures):
    """Learned convolutional feature group.

    Generates features by learning convolutional kernels directly from the
    image using MiniBatchKMeans clustering of image patches. The cluster
    centers serve as representative local patterns and are used as
    convolutional filters.

    Attributes
    ----------
    size : int
        Side length of each learned kernel.
    num_kernels : int or None
        Number of kernels to learn. If ``None``, defaults to ``size ** ndim``.
    num_patches : int or float
        Number of image patches used for learning the kernels.
    exclude_center : bool
        Whether the center pixel is excluded from features.
    """

    def __init__(
        self,
        size: int,
        num_kernels: Optional[int],
        num_patches: Union[int, float] = 1e5,
    ):
        """
        Constructor that configures these features.

        Parameters
        ----------
        size : int
            Filter size
        num_kernels : Optional[int]
            Number of kernels (filters)
        num_patches : Union[int, float]
            Number of patches used for learning the kernels.
        """
        super().__init__(kernels=None)
        self.size = size
        self.num_kernels = num_kernels
        self.num_patches = num_patches

        self.image = None
        self.exclude_center: bool = False

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
        """Return the number of learned convolutional features.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions (unused).

        Returns
        -------
        num : int
            Number of kernels to learn.
        """
        return self.num_kernels

    def learn(self, image):
        """Learn representative convolutional kernels from the image.

        Uses MiniBatchKMeans clustering on image patches to discover
        representative local patterns that serve as convolutional kernels.

        Parameters
        ----------
        image : numpy.ndarray
            Image from which to extract kernels.
        """
        self.kernels = extract_kernels(
            image,
            size=self.size,
            num_kernels=self.num_kernels,
            num_patches=self.num_patches,
        )

    def prepare(self, image, excluded_voxels=None, **kwargs):
        """Prepare the learned correlation feature group for computation.

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

        self.image = image

        super().prepare(image, excluded_voxels, **kwargs)
