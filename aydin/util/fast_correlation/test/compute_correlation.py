import numpy
from numpy.random import rand
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

from aydin.io.datasets import newyork, examples_single


def _normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def compute_correlation_1d(_fun_):
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    kernel = rand(3)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    numpy.testing.assert_array_almost_equal(
        filtered_image[1:-1], scipy_filtered_image[1:-1], decimal=1
    )


def compute_correlation_2d(_fun_, shape=(5, 7)):
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
    hcr = examples_single.royerlab_hcr.get_array().squeeze()
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
    image = examples_single.maitre_mouse.get_array().squeeze()
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
