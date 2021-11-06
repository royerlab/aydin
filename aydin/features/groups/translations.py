from typing import Tuple, Sequence

from scipy.ndimage import shift

from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint


class TranslationFeatures(FeatureGroupBase):
    """
    Translations Feature Group class
    """

    def __init__(self, translations: Sequence[Tuple[int, ...]]):
        super().__init__()
        self.translations = list(translations)
        self.image = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []

        self.kwargs = None

    @property
    def receptive_field_radius(self) -> int:
        radius = max(max(abs(d) for d in t) for t in self.translations)
        return radius

    def num_features(self, ndim: int) -> int:
        return len(self.translations)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self.image = image
        self.excluded_voxels = excluded_voxels
        self.kwargs = kwargs

    def compute_feature(self, index: int, feature):
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
        # Here we cleanup any resource alocated for the last feature computation:
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
