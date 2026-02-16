"""Tests for the RandomFeatures feature group."""

import numpy

from aydin.features.groups.random import RandomFeatures


def test_random_feature_group(normalized_camera_image):
    """Test that random convolutional features produce non-trivial results."""
    # get image:
    image = normalized_camera_image

    # Instantiates DCT features:
    randomf = RandomFeatures(size=5)
    assert randomf.num_features(image.ndim) == 25
    assert randomf.receptive_field_radius == 2

    # Set image:
    randomf.prepare(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(randomf.num_features(image.ndim)):
        randomf.compute_feature(index=index, feature=feature)
        assert (feature != image).any()

    randomf.finish()
