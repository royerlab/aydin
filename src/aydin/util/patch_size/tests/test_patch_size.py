"""Tests for default_patch_size utility."""

import numpy as np

from aydin.util.patch_size.patch_size import default_patch_size


def test_1d_odd_default():
    """Test default odd patch size for a 1D image."""
    image = np.zeros(100)
    result = default_patch_size(image, patch_size=None, odd=True)
    assert result == (17,)


def test_1d_even_default():
    """Test default even patch size for a 1D image."""
    image = np.zeros(100)
    result = default_patch_size(image, patch_size=None, odd=False)
    assert result == (16,)


def test_2d_odd_default():
    """Test default odd patch size for a 2D image."""
    image = np.zeros((64, 64))
    result = default_patch_size(image, patch_size=None, odd=True)
    assert result == (7, 7)


def test_3d_odd_default():
    """Test default odd patch size for a 3D image."""
    image = np.zeros((32, 32, 32))
    result = default_patch_size(image, patch_size=None, odd=True)
    assert result == (5, 5, 5)


def test_4d_odd_default():
    """Test default odd patch size for a 4D image."""
    image = np.zeros((16, 16, 16, 16))
    result = default_patch_size(image, patch_size=None, odd=True)
    assert result == (3, 3, 3, 3)


def test_scalar_broadcast():
    """Test that a scalar patch_size is broadcast to all dimensions."""
    image = np.zeros((64, 64))
    result = default_patch_size(image, patch_size=5)
    assert result == (5, 5)


def test_tuple_passthrough():
    """Test that a tuple patch_size is passed through unchanged."""
    image = np.zeros((64, 64))
    result = default_patch_size(image, patch_size=(3, 5))
    assert result == (3, 5)


def test_clamp_to_half_image():
    """Patch size should be clamped to max(3, shape//2) per dimension."""
    image = np.zeros((8, 8))
    # Default 2D odd = 7, but half of 8 is 4, so should clamp to 4
    result = default_patch_size(image, patch_size=None, odd=True)
    assert all(ps <= max(3, s // 2) for ps, s in zip(result, image.shape))


def test_small_image_clamp():
    """Very small image should clamp patch to 3 minimum."""
    image = np.zeros((4, 4))
    result = default_patch_size(image, patch_size=15)
    assert all(ps <= max(3, s // 2) for ps, s in zip(result, image.shape))
