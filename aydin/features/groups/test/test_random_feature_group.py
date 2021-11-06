import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.random import RandomFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_random_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates DCT features:
    randomf = RandomFeatures(size=5)
    assert randomf.num_features(image.ndim) == 25
    assert randomf.receptive_field_radius == 2

    # Set image:
    randomf.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(randomf.num_features(image.ndim)):
        randomf.compute_feature(index=index, feature=feature)
        assert (feature != image).any()
