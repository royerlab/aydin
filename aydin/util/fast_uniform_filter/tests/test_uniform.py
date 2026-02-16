"""Tests for fast uniform filter implementations."""

import numpy
import pytest
from scipy.ndimage import uniform_filter
from skimage.exposure import rescale_intensity

from aydin.io.datasets import examples_single, newyork, small_newyork
from aydin.util.fast_uniform_filter.numba_cpu_uf import numba_cpu_uniform_filter
from aydin.util.fast_uniform_filter.numba_gpu_uf import numba_gpu_uniform_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.util.fast_uniform_filter.scipy_uf import scipy_uniform_filter


def test_uniform_filter_type_support():
    """Test CPU uniform filter implementations handle various NumPy dtypes."""
    _test_compute_uniform_filter_type_support(scipy_uniform_filter)
    _test_compute_uniform_filter_type_support(parallel_uniform_filter)
    _test_compute_uniform_filter_type_support(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_uniform_filter_type_support_gpu():
    """Test GPU uniform filter handles various NumPy dtypes."""
    _test_compute_uniform_filter_type_support(numba_gpu_uniform_filter)


def test_uniform_filter_different_sizes():
    """Test CPU uniform filter implementations with various kernel sizes."""
    _test_compute_uniform_filter_different_sizes(scipy_uniform_filter)
    _test_compute_uniform_filter_different_sizes(parallel_uniform_filter)
    _test_compute_uniform_filter_different_sizes(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_uniform_filter_different_sizes_gpu():
    """Test GPU uniform filter with various kernel sizes."""
    _test_compute_uniform_filter_different_sizes(numba_gpu_uniform_filter)


def test_compute_uniform_filter_1d():
    """Test CPU uniform filter implementations on 1D data."""
    _test_compute_uniform_filter_1d(scipy_uniform_filter)
    _test_compute_uniform_filter_1d(parallel_uniform_filter)
    _test_compute_uniform_filter_1d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_1d_gpu():
    """Test GPU uniform filter on 1D data."""
    _test_compute_uniform_filter_1d(numba_gpu_uniform_filter)


def test_compute_uniform_filter_2d():
    """Test CPU uniform filter implementations on 2D data."""
    _test_compute_uniform_filter_2d(scipy_uniform_filter)
    _test_compute_uniform_filter_2d(parallel_uniform_filter)
    _test_compute_uniform_filter_2d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_2d_gpu():
    """Test GPU uniform filter on 2D data."""
    _test_compute_uniform_filter_2d(numba_gpu_uniform_filter)


def test_compute_uniform_filter_3d():
    """Test CPU uniform filter implementations on 3D data."""
    _test_compute_uniform_filter_3d(scipy_uniform_filter)
    _test_compute_uniform_filter_3d(parallel_uniform_filter)
    _test_compute_uniform_filter_3d(numba_cpu_uniform_filter)


@pytest.mark.gpu
def test_compute_uniform_filter_3d_gpu():
    """Test GPU uniform filter on 3D data."""
    _test_compute_uniform_filter_3d(numba_gpu_uniform_filter)


def _normalise(image):
    """Rescale image intensity to [0, 1] float32 range."""
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def _test_compute_uniform_filter_type_support(_fun_):
    """Test uniform filter with various numeric dtypes."""
    image = small_newyork()

    _run_test_for_type(_fun_, image.astype(dtype=numpy.float32))
    _run_test_for_type(_fun_, image.astype(dtype=numpy.float16), decimal=1)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint32), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint16), decimal=0)
    _run_test_for_type(_fun_, image.astype(dtype=numpy.uint8), decimal=0)


def _test_compute_uniform_filter_different_sizes(_fun_):
    """Test uniform filter with various kernel sizes and anisotropic sizes."""
    sizes = [1, 2, 3, 4, 5, 7, 8, 16, 32, 128]

    for size in sizes:
        _test_compute_uniform_filter_2d(_fun_, size=size)

    sizes = [(1, 2), (16, 7), (3, 7), (4, 4)]

    for size in sizes:
        _test_compute_uniform_filter_2d(_fun_, size=size)


def _run_test_for_type(_fun_, image, decimal=3, size=3):
    """Run uniform filter and compare against SciPy for a given dtype."""
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
    """Test 1D uniform filter against SciPy reference."""
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_2d(_fun_, size=3):
    """Test 2D uniform filter against SciPy reference."""
    image = _normalise(newyork().astype(numpy.float32))

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_3d(_fun_, size=3):
    """Test 3D uniform filter against SciPy reference."""
    arr = examples_single.royerlab_hcr.get_array()
    if arr is None:
        pytest.skip("royerlab_hcr example could not be loaded")
    hcr = arr.squeeze()
    image = hcr[2, :60, 0 : 0 + 1524, 0 : 0 + 1524]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )


def _test_compute_uniform_filter_4d(_fun_, size=3):
    """Test 4D uniform filter against SciPy reference."""
    arr = examples_single.hyman_hela.get_array()
    if arr is None:
        pytest.skip("hyman_hela example could not be loaded")
    image = arr.squeeze()
    image = image[..., 0:128, 0:128]

    filtered_image = _fun_(image, size=size)
    scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        filtered_image, scipy_filtered_image, decimal=3
    )
