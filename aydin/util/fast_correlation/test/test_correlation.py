import numpy
import pytest
from numpy.random import rand
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

from aydin.io.datasets import small_newyork
from aydin.util.fast_correlation.numba_cpu import numba_cpu_correlate
from aydin.util.fast_correlation.parallel import parallel_correlate
from aydin.util.fast_correlation.test.compute_correlation import (
    compute_correlation_1d,
    compute_correlation_2d,
    compute_correlation_3d,
    compute_correlation_4d,
    compute_correlation_5d,
    compute_correlation_6d,
)


def test_correlation_type_support():
    _test_compute_correlation_type_support(numba_cpu_correlate)
    _test_compute_correlation_type_support(parallel_correlate)


def test_correlation_different_sizes():
    _test_compute_correlation_different_sizes(numba_cpu_correlate)
    _test_compute_correlation_different_sizes(parallel_correlate)


@pytest.mark.parametrize(
    "method",
    [
        compute_correlation_1d,
        compute_correlation_2d,
        compute_correlation_3d,
        compute_correlation_4d,
        compute_correlation_5d,
        compute_correlation_6d,
    ],
)
def test_compute_correlation(method):
    method(numba_cpu_correlate)
    method(parallel_correlate)


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
        compute_correlation_2d(_fun_, shape=(size, size))

    shapes = [(1, 3), (15, 7), (3, 7), (5, 1)]

    for shape in shapes:
        compute_correlation_2d(_fun_, shape=shape)


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
