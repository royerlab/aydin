"""Tests for the salt-and-pepper noise suppression transform."""

# flake8: noqa
import numpy
from numpy.linalg import norm
from scipy.ndimage import median_filter
from skimage.util import random_noise

from aydin.io.datasets import camera, newyork, normalise
from aydin.it.transforms.salt_pepper import SaltPepperTransform


def test_suppress_fixed_background_real():
    """Test salt-and-pepper correction reduces error below median filter baseline."""
    image = normalise(newyork())
    noisy = random_noise(image, mode="s&p", amount=0.03, rng=0, clip=False)

    bpc = SaltPepperTransform(threshold=0.15)

    corrected = bpc.preprocess(noisy)

    median = median_filter(image, size=3)

    error0 = numpy.abs(median - image).mean()
    error = numpy.abs(corrected - image).mean()

    print(f"Error noisy = {error0}")
    print(f"Error = {error}")
    assert error < 0.03
    assert error0 > error


def test_salt_and_pepper_3d():
    """Test salt and pepper transform on 3D volume."""
    # Use gradient data instead of binary blobs for realistic grayscale input
    # (SaltPepperTransform doesn't work well with binary images)
    x = numpy.linspace(0.1, 0.9, 32)
    image = (
        x[:, numpy.newaxis, numpy.newaxis]
        + x[numpy.newaxis, :, numpy.newaxis] * 0.3
        + x[numpy.newaxis, numpy.newaxis, :] * 0.2
    )
    image = normalise(image.astype(numpy.float32))
    noisy = random_noise(image, mode="s&p", amount=0.03, rng=0, clip=False).astype(
        numpy.float32
    )

    bpc = SaltPepperTransform(threshold=0.15)

    corrected = bpc.preprocess(noisy)

    median = median_filter(image, size=3)

    error0 = numpy.abs(median - image).mean()
    error = numpy.abs(corrected - image).mean()

    print(f"Error noisy = {error0}")
    print(f"Error = {error}")

    assert corrected.shape == noisy.shape
    assert error < 0.03
    assert error0 > error
