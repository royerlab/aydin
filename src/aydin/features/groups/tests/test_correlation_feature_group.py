"""Tests for the CorrelationFeatures feature group."""

import numpy
from scipy.ndimage import convolve

from aydin.features.groups.correlation import CorrelationFeatures


def test_convolutional_feature_group(normalized_camera_image):
    """Test that correlation features match scipy convolve output."""
    # get image:
    image = normalized_camera_image

    # Instantiates convolutional features:
    ones = numpy.ones(shape=(3, 3))
    twos = 2 * numpy.ones(shape=(3, 3))
    convolutions = CorrelationFeatures(kernels=[ones, twos])
    assert convolutions.num_features(image.ndim) == 2

    # Check receptive field radius:
    assert convolutions.receptive_field_radius == 1

    # Set image:
    convolutions.prepare(image, [])

    # compute features and check their validity:
    feature = numpy.empty_like(image)

    # Compute first convolution:
    convolutions.compute_feature(index=0, feature=feature)
    assert (feature == convolve(image, weights=ones)).all()

    # Compute second convolution:
    convolutions.compute_feature(index=1, feature=feature)
    assert (feature == convolve(image, weights=twos)).all()

    convolutions.finish()


def test_correlation_feature_with_excluded_voxels(normalized_camera_image):
    """Test correlation features with excluded voxels (blind-spot path)."""
    image = normalized_camera_image

    ones = numpy.ones(shape=(3, 3))
    convolutions = CorrelationFeatures(kernels=[ones])
    # Prepare with excluded center voxel (the blind-spot case):
    convolutions.prepare(image, excluded_voxels=[(0, 0)])
    feature = numpy.empty_like(image)
    convolutions.compute_feature(index=0, feature=feature)
    # The center pixel should NOT contribute (blind-spot):
    assert feature is not None
    assert feature.shape == image.shape

    # Verify it differs from the non-excluded version:
    convolutions2 = CorrelationFeatures(kernels=[numpy.ones(shape=(3, 3))])
    convolutions2.prepare(image, excluded_voxels=[])
    feature2 = numpy.empty_like(image)
    convolutions2.compute_feature(index=0, feature=feature2)
    assert not numpy.array_equal(feature, feature2)
