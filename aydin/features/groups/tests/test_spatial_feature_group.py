"""Tests for the SpatialFeatures feature group."""

import numpy

from aydin.features.groups.spatial import SpatialFeatures


def test_spatial_feature_group(normalized_camera_image):
    """Test that spatial features encode correct coordinate axes."""
    # get image:
    image = normalized_camera_image

    # Instantiates spatial features:
    spatial = SpatialFeatures()
    assert spatial.num_features(image.ndim) == image.ndim

    # Check receptive field radius:
    assert spatial.receptive_field_radius == 0

    # Set image:
    spatial.prepare(image)

    # compute features and check their validity:
    feature_y = numpy.empty_like(image)
    spatial.compute_feature(index=0, feature=feature_y)
    assert feature_y[0, 0] == feature_y[0, 511]
    assert feature_y[0, 0] < feature_y[511, 0]

    feature_x = numpy.empty_like(image)
    spatial.compute_feature(index=1, feature=feature_x)
    assert feature_x[0, 0] < feature_x[0, 511]
    assert feature_x[0, 0] == feature_x[511, 0]

    spatial.finish()
