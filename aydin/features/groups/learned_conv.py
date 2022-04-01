from typing import Union, Optional

from aydin.features.groups.convolutional import ConvolutionalFeatures
from aydin.features.groups.extract_kernels import extract_kernels


class LearnedConvolutionalFeatures(ConvolutionalFeatures):
    """
    Learned Convolutions Feature Group class

    Generates features by learning convolutional filters on the basis of the
    image itself.
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
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        return self.num_kernels

    def learn(self, image):
        self.kernels = extract_kernels(
            image,
            size=self.size,
            num_kernels=self.num_kernels,
            num_patches=self.num_patches,
        )

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image

        super().prepare(image, excluded_voxels, **kwargs)
