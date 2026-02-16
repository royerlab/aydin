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
