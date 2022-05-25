import numpy
from numpy.random import rand
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

from aydin.io.datasets import newyork, examples_single, small_newyork
from aydin.util.fast_correlation.numba_cpu import numba_cpu_correlate
from aydin.util.fast_correlation.parallel import parallel_correlate


def test_correlation_type_support():
    _test_compute_correlation_type_support(numba_cpu_correlate)
    _test_compute_correlation_type_support(parallel_correlate)


def test_correlation_different_sizes():
    _test_compute_correlation_different_sizes(numba_cpu_correlate)
    _test_compute_correlation_different_sizes(parallel_correlate)


def test_compute_correlation_1d():
    _test_compute_correlation_1d(numba_cpu_correlate)
    _test_compute_correlation_1d(parallel_correlate)


def test_compute_correlation_2d():
    _test_compute_correlation_2d(numba_cpu_correlate)
    _test_compute_correlation_2d(parallel_correlate)


def test_compute_correlation_3d():
    _test_compute_correlation_3d(numba_cpu_correlate)
    _test_compute_correlation_3d(parallel_correlate)


def test_compute_correlation_4d():
    _test_compute_correlation_4d(numba_cpu_correlate)
    _test_compute_correlation_4d(parallel_correlate)


def test_compute_correlation_5d():
    _test_compute_correlation_5d(numba_cpu_correlate)
    _test_compute_correlation_5d(parallel_correlate)


def test_compute_correlation_6d():
    _test_compute_correlation_6d(numba_cpu_correlate)
    _test_compute_correlation_6d(parallel_correlate)


def _normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def _test_compute_correlation_type_support(_fun_):
    image = small_newyork()

    _run_test_for_type(_fun_, image.astype(dtype=numpy.float32))
    _run_test_for_type(_fun_, image.astype(dtype=numpy.float16), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint32), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint16), decimal=0)
    # _run_test_for_type(_fun_, image.astype(dtype=numpy.uint8), decimal=0)


def _test_compute_correlation_different_sizes(_fun_):
    sizes = [1, 3, 5, 7, 17, 31]

    for size in sizes:
        _test_compute_correlation_2d(_fun_, shape=(size, size))

    shapes = [(1, 3), (15, 7), (3, 7), (5, 1)]

    for shape in shapes:
        _test_compute_correlation_2d(_fun_, shape=shape)


def _run_test_for_type(_fun_, image, decimal=3):

    kernel = rand(3, 5)

    scipy_filtered_image = correlate(
        image.astype(dtype=numpy.float32, copy=False), weights=kernel
    )

    filtered_image = _fun_(image, kernel=kernel)

    filtered_image = filtered_image.astype(dtype=numpy.float32, copy=False)
    scipy_filtered_image = scipy_filtered_image.astype(dtype=numpy.float32, copy=False)

    filtered_image = filtered_image[1:-1, 2:-2]
    scipy_filtered_image = scipy_filtered_image[1:-1, 2:-2]

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=decimal
    )


def _test_compute_correlation_1d(_fun_):
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    kernel = rand(3)

    filtered_image = _fun_(image, kernel=kernel)
    scipy_filtered_image = correlate(image, weights=kernel)

    numpy.testing.assert_array_almost_equal(
        filtered_image[1:-1], scipy_filtered_image[1:-1], decimal=1
    )


def _test_compute_correlation_2d(_fun_, shape=(5, 7)):
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


def _test_compute_correlation_3d(_fun_, shape=(3, 5, 7)):
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


def _test_compute_correlation_4d(_fun_, shape=(3, 5, 7, 9)):
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


def _test_compute_correlation_5d(_fun_, shape=(3, 1, 3, 1, 3)):
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


def _test_compute_correlation_6d(_fun_, shape=(1, 3, 1, 3, 1, 3)):
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
