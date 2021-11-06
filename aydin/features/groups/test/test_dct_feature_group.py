import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.dct import DCTFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
    )


def test_dct_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates DCT features:
    convolutions = DCTFeatures(size=5, max_freq=2, power=1)
    assert convolutions.num_features(image.ndim) == 25
    assert convolutions.receptive_field_radius == 2

    # Set image:
    convolutions.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(convolutions.num_features(image.ndim)):
        convolutions.compute_feature(index=index, feature=feature)
        assert (feature != image).any()
