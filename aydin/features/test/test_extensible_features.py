import numpy
from skimage.exposure import rescale_intensity

from aydin.features.extensible_features import ExtensibleFeatureGenerator
from aydin.features.groups.median import MedianFeatures
from aydin.features.groups.translations import TranslationFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_extensible_features_basics():
    image = n(camera().astype(numpy.float32))

    # feature generator requires images in 'standard' form: BCTZYX
    image = image[numpy.newaxis, numpy.newaxis, ...]

    feature_gen = ExtensibleFeatureGenerator()

    vectors = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    translations = TranslationFeatures(translations=vectors)
    feature_gen.add_feature_group(translations)

    radii = [1, 2, 4, 8]
    medians = MedianFeatures(radii=radii)
    feature_gen.add_feature_group(medians)

    features = feature_gen.compute(image, exclude_center_feature=True)

    assert feature_gen.get_num_features(image.ndim - 2) == 8 + 4
    assert feature_gen.get_receptive_field_radius() == 8

    assert features is not None
    assert features.shape == (1, 512, 512, 12)
