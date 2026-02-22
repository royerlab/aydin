"""Helper functions that validate correlation implementations against SciPy."""

import numpy
import pytest
from numpy.random import rand
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

from aydin.io.datasets import examples_single, newyork


def _normalise(image):
    """Rescale image intensity to [0, 1] float32 range."""
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def compute_correlation_1d(_fun_):
    """Validate a 1D correlation function against SciPy's correlate."""
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    kernel = rand(3)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    numpy.testing.assert_array_almost_equal(
        filtered_image[1:-1], scipy_filtered_image[1:-1], decimal=1
    )


def compute_correlation_2d(_fun_, shape=(5, 7)):
    """Validate a 2D correlation function against SciPy's correlate."""
    image = _normalise(newyork())
    image = image.astype(numpy.float32)[0:731, 0:897]

    kernel = rand(*shape)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    filtered_image = filtered_image[31:-31, 31:-31]
    scipy_filtered_image = scipy_filtered_image[31:-31, 31:-31]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=0
    )


def compute_correlation_3d(_fun_, shape=(3, 5, 7)):
    """Validate a 3D correlation function against SciPy's correlate."""
    arr = examples_single.royerlab_hcr.get_array()
    if arr is None:
        pytest.skip("royerlab_hcr example could not be loaded")
    hcr = arr.squeeze()
    image = hcr[:60, 2, 0 : 0 + 1524, 0 : 0 + 1524]
    image = image.astype(numpy.float32)

    kernel = rand(*shape)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    filtered_image = filtered_image[1:-1, 2:-2, 3:-3]
    scipy_filtered_image = scipy_filtered_image[1:-1, 2:-2, 3:-3]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=1
    )


def compute_correlation_4d(_fun_, shape=(3, 5, 7, 9)):
    """Validate a 4D correlation function against SciPy's correlate."""
    arr = examples_single.maitre_mouse.get_array()
    if arr is None:
        pytest.skip("maitre_mouse example could not be loaded")
    image = arr.squeeze()
    image = image[..., 0:64, 0:64]
    image = image.astype(numpy.float32)

    kernel = rand(*shape)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    filtered_image = filtered_image[1:-1, 2:-2, 3:-3, 4:-4]
    scipy_filtered_image = scipy_filtered_image[1:-1, 2:-2, 3:-3, 4:-4]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=1
    )


def compute_correlation_5d(_fun_, shape=(3, 1, 3, 1, 3)):
    """Validate a 5D correlation function against SciPy's correlate."""
    image = rand(7, 6, 5, 7, 3)
    image = image.astype(numpy.float32)

    kernel = rand(*shape)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    filtered_image = filtered_image[1:-1, :, 1:-1, :, 1:-1]
    scipy_filtered_image = scipy_filtered_image[1:-1, :, 1:-1, :, 1:-1]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=1
    )


def compute_correlation_6d(_fun_, shape=(1, 3, 1, 3, 1, 3)):
    """Validate a 6D correlation function against SciPy's correlate."""
    image = rand(7, 8, 5, 6, 3, 5)
    image = image.astype(numpy.float32)

    kernel = rand(*shape)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    filtered_image = filtered_image[:, 1:-1, :, 1:-1, :, 1:-1]
    scipy_filtered_image = scipy_filtered_image[:, 1:-1, :, 1:-1, :, 1:-1]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=1
    )
