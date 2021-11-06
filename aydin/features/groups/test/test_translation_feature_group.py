import numpy
from scipy.ndimage import shift
from skimage.exposure import rescale_intensity

from aydin.features.groups.translations import TranslationFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_translation_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates translation features:
    vectors = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    translations = TranslationFeatures(translations=vectors)
    assert translations.num_features(image.ndim) == 8

    # Check receptive field radius:
    assert translations.receptive_field_radius == 1

    # Set image:
    translations.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)
    for index in range(translations.num_features(image.ndim)):
        translations.compute_feature(index=index, feature=feature)
        vector = vectors[index]
        translated = shift(
            image,
            shift=list(vector),
            output=feature,
            order=0,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
        assert (feature == translated).all()
