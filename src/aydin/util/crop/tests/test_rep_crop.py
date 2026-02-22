"""Tests for the representative crop utility."""

# flake8: noqa

import numpy
import pytest

from aydin.io.datasets import newyork
from aydin.util.crop.demo.demo_rep_crop import demo_representative_crop
from aydin.util.crop.demo.demo_sf_rep_crop import demo_super_fast_representative_crop
from aydin.util.crop.rep_crop import _normalise, _rescale, representative_crop


def test_representative_crop():
    """Test representative crop demo runs without errors."""
    newyork_image = newyork()
    demo_representative_crop(newyork_image, display=False)


def test_super_fast_representative_crop():
    """Test super-fast representative crop demo runs without errors."""
    newyork_image = newyork()
    demo_super_fast_representative_crop(newyork_image, display=False)


def test_representative_crop_small_batch_dimension():
    """Regression test for GitHub issue #207.

    A ZeroDivisionError occurred when the image had a small batch dimension
    (e.g., shape (2, 84, 580, 576)) because the granularity for that
    dimension was computed as 0 (cs // granularity_factor where cs < factor).
    """
    numpy.random.seed(42)
    # The exact shape from the bug report
    image = numpy.random.rand(2, 84, 580, 576).astype(numpy.float32)
    crop = representative_crop(image, crop_size=int(1e6), favour_odd_lengths=True)
    assert crop is not None
    assert crop.ndim == 4


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 64),
        (3, 3, 64, 64),
        (2, 2, 2, 32, 32),
        (1, 1, 128, 128),
    ],
)
def test_representative_crop_small_leading_dimensions(shape):
    """Ensure representative_crop handles images with small leading
    dimensions (common for batch/channel axes) without errors."""
    numpy.random.seed(42)
    image = numpy.random.rand(*shape).astype(numpy.float32)
    crop = representative_crop(image, crop_size=int(1e5))
    assert crop is not None
    assert crop.ndim == len(shape)


def test_representative_crop_dimension_equals_crop():
    """Test that no error occurs when a dimension exactly equals the
    cropped size (the cs == s edge case that caused the original bug)."""
    numpy.random.seed(42)
    # Small image where crop_size is larger than the image itself
    image = numpy.random.rand(32, 32).astype(numpy.float32)
    crop = representative_crop(image, crop_size=32 * 32 + 1)
    assert crop is not None
    # When image is already small enough, the whole image is returned
    assert crop.shape == image.shape


# --- Mode tests ---


def test_representative_crop_contrast_mode():
    """Test contrast scoring mode."""
    numpy.random.seed(42)
    image = numpy.random.rand(128, 128).astype(numpy.float32)
    crop = representative_crop(image, mode='contrast', crop_size=64 * 64)
    assert crop is not None
    assert crop.ndim == 2


def test_representative_crop_sobel_mode():
    """Test sobel scoring mode."""
    numpy.random.seed(42)
    image = numpy.random.rand(128, 128).astype(numpy.float32)
    crop = representative_crop(image, mode='sobel', crop_size=64 * 64)
    assert crop is not None
    assert crop.ndim == 2


# --- _normalise / _rescale tests ---


def test_normalise_normal_case():
    """Normalised image should be in [0, 1]."""
    image = numpy.array([10.0, 20.0, 30.0], dtype=numpy.float32)
    result = _normalise(image)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalise_constant_image():
    """Constant image should produce all zeros after normalisation."""
    image = numpy.full(10, 5.0, dtype=numpy.float32)
    result = _normalise(image)
    assert numpy.all(result == 0.0)


def test_rescale_values():
    """_rescale should map min to 0 and max to 1."""
    x = numpy.array([2.0, 4.0, 6.0], dtype=numpy.float32)
    result = _rescale(x, numpy.float32(2.0), numpy.float32(6.0))
    numpy.testing.assert_almost_equal(result, [0.0, 0.5, 1.0])
    assert result.dtype == numpy.float32


# --- return_slice and 3D tests ---


def test_representative_crop_return_slice():
    """return_slice=True should return (crop, slice_tuple)."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    result = representative_crop(image, crop_size=32 * 32, return_slice=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    crop, slices = result
    assert crop.ndim == 2
    assert len(slices) == 2


def test_representative_crop_3d():
    """Should work on 3D images."""
    numpy.random.seed(42)
    image = numpy.random.rand(16, 64, 64).astype(numpy.float32)
    crop = representative_crop(image, crop_size=8 * 32 * 32)
    assert crop is not None
    assert crop.ndim == 3
