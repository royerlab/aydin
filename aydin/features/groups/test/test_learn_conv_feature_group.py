import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.learned_conv import LearnedConvolutionalFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_learned_conv_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates DCT features:
    lconv = LearnedConvolutionalFeatures(size=5, num_kernels=30)
    assert lconv.num_features(image.ndim) == 30
    assert lconv.receptive_field_radius == 2

    # Set image:
    lconv.prepare(image)

    # Learn
    lconv.learn(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(lconv.num_features(image.ndim)):
        lconv.compute_feature(index=index, feature=feature)
        assert (feature != image).any()
