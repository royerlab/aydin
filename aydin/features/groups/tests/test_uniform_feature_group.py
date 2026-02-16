"""Tests for the UniformFeatures feature group."""

import numpy

from aydin.features.groups.uniform import UniformFeatures


def test_uniform_feature_group(normalized_camera_image):
    """Test uniform feature generation count, receptive field, and computation."""
    # get image:
    image = normalized_camera_image

    # Instantiates median features:
    uniform = UniformFeatures(
        include_corner_features=True, include_scale_one=True, include_fine_features=True
    )

    assert uniform.num_features(image.ndim) == 65

    # Check receptive field radius:
    assert uniform.receptive_field_radius == 3070

    # Set image:
    uniform.prepare(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)
    for index in range(uniform.num_features(image.ndim)):
        uniform.compute_feature(index=index, feature=feature)
        assert (feature != image).any()

    uniform.finish()
