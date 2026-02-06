"""Tests for the CorrelationFeatures feature group."""

import numpy
from scipy.ndimage import convolve
from skimage.exposure import rescale_intensity

from aydin.features.groups.correlation import CorrelationFeatures
from aydin.io.datasets import camera


def n(image):
    """Normalize image to float32 in [0, 1] range."""
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_convolutional_feature_group():
    """Test that correlation features match scipy convolve output."""
    # get image:
    image = n(camera().astype(numpy.float32))

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
