"""Tests for the DCTFeatures feature group."""

import numpy

from aydin.features.groups.dct import DCTFeatures


def test_dct_feature_group(normalized_camera_image):
    """Test that DCT features produce non-trivial results."""
    # get image:
    image = normalized_camera_image

    # Instantiates DCT features:
    convolutions = DCTFeatures(size=5, max_freq=2, power=1)
    assert convolutions.num_features(image.ndim) == 25
    assert convolutions.receptive_field_radius == 2

    # Set image:
    convolutions.prepare(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(convolutions.num_features(image.ndim)):
        convolutions.compute_feature(index=index, feature=feature)
        assert (feature != image).any()

    convolutions.finish()
