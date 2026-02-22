"""Tests for the MedianFeatures feature group."""

import numpy
from scipy.ndimage import median_filter

from aydin.features.groups.median import MedianFeatures


def test_median_feature_group(normalized_camera_image):
    """Test median features match scipy median_filter output."""
    # get image:
    image = normalized_camera_image

    # Instantiates median features:
    radii = [1, 2, 4, 8]
    medians = MedianFeatures(radii=radii)
    assert medians.num_features(image.ndim) == 4

    # Check receptive field radius:
    assert medians.receptive_field_radius == 8

    # Set image:
    medians.prepare(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)
    for index in range(medians.num_features(image.ndim)):
        medians.compute_feature(index=index, feature=feature)
        radius = radii[index]
        assert (feature == median_filter(image, size=2 * radius + 1)).all()

    medians.finish()
