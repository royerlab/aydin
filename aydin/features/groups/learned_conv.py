from typing import Union, Optional

from aydin.features.groups.convolutional import ConvolutionalFeatures
from aydin.features.groups.extract_kernels import extract_kernels


class LearnedConvolutionalFeatures(ConvolutionalFeatures):
    """
    Learned Convolutions Feature Group class
    """

    def __init__(
        self,
        size: int,
        num_kernels: Optional[int],
        num_patches: Union[int, float] = 1e5,
    ):
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
