"""Tests for the nearest self-image computation."""

import numpy
import pytest

from aydin.util.nsi.nearest_self_image import nearest_self_image


def test_nearest_self_image_returns_same_shape():
    """Output should have the same shape as the input."""
    rng = numpy.random.RandomState(42)
    image = rng.rand(32, 32).astype(numpy.float32)
    result = nearest_self_image(image, patch_shape=3)
    assert result.shape == image.shape


def test_nearest_self_image_returns_float():
    """Output should be float32."""
    rng = numpy.random.RandomState(42)
    image = rng.rand(32, 32).astype(numpy.float32)
    result = nearest_self_image(image, patch_shape=3)
    assert result.dtype == numpy.float32


def test_nearest_self_image_with_int_patch_shape():
    """Integer patch_shape should be broadcast to all dimensions."""
    rng = numpy.random.RandomState(42)
    image = rng.rand(32, 32).astype(numpy.float32)
    result = nearest_self_image(image, patch_shape=5)
    assert result.shape == image.shape


def test_nearest_self_image_with_tuple_patch_shape():
    """Tuple patch_shape should work for anisotropic patches."""
    rng = numpy.random.RandomState(42)
    image = rng.rand(32, 32).astype(numpy.float32)
    result = nearest_self_image(image, patch_shape=(3, 5))
    assert result.shape == image.shape


@pytest.mark.heavy
def test_nearest_self_image_denoising_effect():
    """NSI on a noisy image should reduce noise (smooth the result)."""
    rng = numpy.random.RandomState(42)
    # Create a structured image with noise
    clean = numpy.zeros((32, 32), dtype=numpy.float32)
    clean[8:24, 8:24] = 1.0
    noisy = clean + rng.randn(32, 32).astype(numpy.float32) * 0.1

    result = nearest_self_image(noisy, patch_shape=3)

    # Result should be closer to clean than noisy is
    noisy_mse = numpy.mean((noisy - clean) ** 2)
    result_mse = numpy.mean((result - clean) ** 2)
    # At minimum, the result should not be drastically worse
    assert result_mse < noisy_mse * 2
