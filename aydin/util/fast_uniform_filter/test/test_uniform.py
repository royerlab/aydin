import numpy
import pytest
from scipy.ndimage import uniform_filter
from skimage.exposure import rescale_intensity

from aydin.util.fast_uniform_filter.numba_cpu_uf import numba_cpu_uniform_filter
from aydin.util.fast_uniform_filter.numba_gpu_uf import numba_gpu_uniform_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.util.fast_uniform_filter.scipy_uf import scipy_uniform_filter
from aydin.io.datasets import newyork, examples_single


def test_uniform_filter_type_support():
    _test_compute_uniform_filter_type_support(scipy_uniform_filter)
    _test_compute_uniform_filter_type_support(parallel_uniform_filter)
    _test_compute_uniform_filter_type_support(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_uniform_filter_type_support_gpu():
    _test_compute_uniform_filter_type_support(numba_gpu_uniform_filter)


def test_uniform_filter_different_sizes():
    _test_compute_uniform_filter_different_sizes(scipy_uniform_filter)
    _test_compute_uniform_filter_different_sizes(parallel_uniform_filter)
    _test_compute_uniform_filter_different_sizes(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_uniform_filter_different_sizes_gpu():
    _test_compute_uniform_filter_different_sizes(numba_gpu_uniform_filter)


def test_compute_uniform_filter_1d():
    _test_compute_uniform_filter_1d(scipy_uniform_filter)
    _test_compute_uniform_filter_1d(parallel_uniform_filter)
    _test_compute_uniform_filter_1d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_1d_gpu():
    _test_compute_uniform_filter_1d(numba_gpu_uniform_filter)


def test_compute_uniform_filter_2d():
    _test_compute_uniform_filter_2d(scipy_uniform_filter)
    _test_compute_uniform_filter_2d(parallel_uniform_filter)
    _test_compute_uniform_filter_2d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_2d_gpu():
    _test_compute_uniform_filter_2d(numba_gpu_uniform_filter)


def test_compute_uniform_filter_3d():
    _test_compute_uniform_filter_3d(scipy_uniform_filter)
    _test_compute_uniform_filter_3d(parallel_uniform_filter)
    _test_compute_uniform_filter_3d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_3d_gpu():
    _test_compute_uniform_filter_3d(numba_gpu_uniform_filter)


def _normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def _test_compute_uniform_filter_type_support(_fun_):
    image = newyork()

    _run_test_for_type(_fun_, image.astype(dtype=numpy.float32))
    _run_test_for_type(_fun_, image.astype(dtype=numpy.float16), decimal=1)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint32), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint16), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint8), decimal=0)


def _test_compute_uniform_filter_different_sizes(_fun_):
    sizes = [1, 2, 3, 4, 5, 7, 8, 16, 32, 128]

    for size in sizes:
        _test_compute_uniform_filter_2d(_fun_, size=size)

    sizes = [(1, 2), (16, 7), (3, 7), (4, 4)]

    for size in sizes:
        _test_compute_uniform_filter_2d(_fun_, size=size)


def _run_test_for_type(_fun_, image, decimal=3, size=3):
    filtered_image = _fun_(image, size=size)
    assert filtered_image.dtype == image.dtype

    filtered_image = filtered_image.astype(dtype=numpy.float32, copy=False)
    scipy_filtered_image = uniform_filter(
        image.astype(dtype=numpy.float32, copy=False), size=size, mode="nearest"
    ).astype(dtype=numpy.float32, copy=False)
    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=decimal
    )


def _test_compute_uniform_filter_1d(_fun_, size=3):
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_2d(_fun_, size=3):
    image = _normalise(newyork().astype(numpy.float32))

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_3d(_fun_, size=3):
    islet = examples_single.royerlab_hcr.get_array().squeeze()
    image = islet[2, :60, 0 : 0 + 1524, 0 : 0 + 1524]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_4d(_fun_, size=3):
    image = examples_single.hyman_hela.get_array().squeeze()
    image = image[..., 0:128, 0:128]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )
