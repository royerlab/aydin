"""Tests for the super-fast representative crop extraction."""

import numpy

from aydin.util.crop.super_fast_rep_crop import super_fast_representative_crop


def _make_test_image(shape=(256, 256), seed=42):
    """Create a test image with some structure."""
    rng = numpy.random.RandomState(seed)
    image = rng.rand(*shape).astype(numpy.float32)
    # Add a bright region so there's a 'most representative' crop
    cx, cy = shape[0] // 2, shape[1] // 2
    image[cx - 20 : cx + 20, cy - 20 : cy + 20] += 2.0
    return image


def test_super_fast_crop_returns_array():
    """Should return a cropped numpy array."""
    image = _make_test_image()
    crop = super_fast_representative_crop(image, crop_size=64 * 64)
    assert isinstance(crop, numpy.ndarray)
    assert crop.size > 0
    assert crop.size <= image.size


def test_super_fast_crop_return_slice():
    """When return_slice=True, returns (crop, slice_tuple)."""
    image = _make_test_image()
    result = super_fast_representative_crop(image, crop_size=64 * 64, return_slice=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    crop, slice_tuple = result
    assert isinstance(crop, numpy.ndarray)
    assert len(slice_tuple) == image.ndim

    # Slice should produce the same crop
    numpy.testing.assert_array_equal(crop, image[slice_tuple])


def test_super_fast_crop_slice_within_bounds():
    """Returned slice should be within image bounds."""
    image = _make_test_image(shape=(128, 128))
    _, slice_tuple = super_fast_representative_crop(
        image, crop_size=32 * 32, return_slice=True
    )
    for sl, s in zip(slice_tuple, image.shape):
        assert sl.start >= 0
        assert sl.stop <= s


def test_super_fast_crop_3d():
    """Should work on 3D images."""
    rng = numpy.random.RandomState(42)
    image = rng.rand(64, 64, 64).astype(numpy.float32)
    crop = super_fast_representative_crop(image, crop_size=16 * 16 * 16)
    assert isinstance(crop, numpy.ndarray)
    assert crop.ndim == 3
    assert crop.size > 0


def test_super_fast_crop_systematic_mode():
    """Should work with systematic search mode."""
    image = _make_test_image()
    crop = super_fast_representative_crop(
        image, crop_size=64 * 64, search_mode='systematic'
    )
    assert isinstance(crop, numpy.ndarray)
    assert crop.size > 0
