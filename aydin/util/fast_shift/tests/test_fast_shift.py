"""Tests for the fast array shift utility."""

import numpy
import pytest
from scipy.ndimage import shift
from skimage.exposure import rescale_intensity

from aydin.io.datasets import examples_single, newyork
from aydin.util.fast_shift.fast_shift import fast_shift


def _normalise(image):
    """Rescale image intensity to [0, 1] float32 range."""
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
    )


@pytest.mark.parametrize(
    "image_dtype, decimal",
    [
        (numpy.float32, 3),
        (numpy.float16, 1),
        (numpy.uint32, 0),
        (numpy.uint16, 0),
        (numpy.uint8, 0),
    ],
)
def test_fast_shift_filter_type_support(image_dtype, decimal, _shift=(-1, 3)):
    """Test fast_shift preserves dtype and matches SciPy for various types."""
    image = newyork()
    shifted_image = fast_shift(image, shift=_shift)
    assert shifted_image.dtype == image.dtype

    shifted_image = shifted_image.astype(dtype=numpy.float32, copy=False)
    scipy_shifted_image = shift(
        image.astype(dtype=numpy.float32, copy=False), shift=_shift, mode="constant"
    ).astype(dtype=numpy.float32, copy=False)
    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=decimal
    )


def test_compute_uniform_filter_different_sizes():
    """Test fast_shift with various 2D shift vectors."""
    shifts = [(1, 2), (16, 7), (3, 7), (4, 4)]

    for current_shift in shifts:
        test_fast_shift_filter_2d(_shift=current_shift)


def test_fast_shift_filter_1d(_shift=(-1,)):
    """Test fast_shift on a 1D array matches SciPy's shift."""
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_2d(_shift=(-1, 3)):
    """Test fast_shift on a 2D image matches SciPy's shift."""
    image = _normalise(newyork().astype(numpy.float32))

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_3d(_shift=(-1, 3, -7)):
    """Test fast_shift on a 3D volume matches SciPy's shift."""
    arr = examples_single.royerlab_hcr.get_array()
    if arr is None:
        pytest.skip("royerlab_hcr example could not be loaded")
    islet = arr.squeeze()
    image = islet[2, :60, 0:256, 0:256]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_4d(_shift=(-1, 3, -7, +13)):
    """Test fast_shift on a 4D array matches SciPy's shift."""
    arr = examples_single.maitre_mouse.get_array()
    if arr is None:
        pytest.skip("maitre_mouse example could not be loaded")
    image = arr.squeeze()
    image = image[0:10, 0:10, 0:128, 0:128]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )
