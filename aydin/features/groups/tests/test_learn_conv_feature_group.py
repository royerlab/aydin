"""Tests for the LearnedCorrelationFeatures feature group."""

import numpy

from aydin.features.groups.learned_conv import LearnedCorrelationFeatures


def test_learned_conv_feature_group(normalized_camera_image):
    """Test that learned convolutional features produce non-trivial results."""
    # get image:
    image = normalized_camera_image

    # Instantiates DCT features:
    lconv = LearnedCorrelationFeatures(size=5, num_kernels=30)
    assert lconv.num_features(image.ndim) == 30
    assert lconv.receptive_field_radius == 2

    # Set image:
    lconv.prepare(image)

    # Learn
    lconv.learn(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(lconv.num_features(image.ndim)):
        lconv.compute_feature(index=index, feature=feature)
        assert (feature != image).any()

    lconv.finish()
