"""Tests for the n-dimensional denoising extension decorator."""

# flake8: noqa
import numpy
import pytest
from scipy.ndimage import gaussian_filter

from aydin.util.denoise_nd.denoise_nd import extend_nd


def test_denoise_nd():
    """Test extend_nd decorator extends a 2D-only function to other dimensions."""

    # raw function that only supports 2D images:
    def function(image, sigma):
        """Apply Gaussian filter only to 2D images, raise on other dims."""
        if image.ndim != 2:
            raise RuntimeError("Function only supports arrays of dimensions 2")
        return gaussian_filter(image, sigma)

    # extended function that supports all dimension (with all caveats associated to how we actually do this extension...)
    @extend_nd(available_dims=[2])
    def extended_function(image, sigma):
        """Gaussian filter extended to arbitrary dimensions via extend_nd."""
        return function(image, sigma)

    # Wrongly extended function: we pretend that it can do dim 1 when in fact it can't!
    @extend_nd(available_dims=[1, 2])
    def wrongly_extended_function(image, sigma):
        """Incorrectly claims 1D support to test error propagation."""
        return function(image, sigma)

    # Test 1D image - raw function should raise RuntimeError
    image_1d = numpy.zeros((32,))
    image_1d[16] = 1

    with pytest.raises(RuntimeError):
        function(image_1d, sigma=1)

    # extended_function should handle 1D by extending to 2D internally
    # Note: returns shape (1, 32) because it adds a dimension for processing
    result = extended_function(image_1d, sigma=1)
    assert result is not None
    assert result.size == image_1d.size  # Same number of elements
    assert result.squeeze().shape == image_1d.shape  # Can squeeze back to original

    # wrongly_extended_function claims to support 1D but underlying function doesn't
    with pytest.raises(RuntimeError):
        wrongly_extended_function(image_1d, sigma=1)

    # Test 3D image
    image_3d = numpy.zeros((32, 5, 64))
    image_3d[16, 2, 32] = 1

    with pytest.raises(RuntimeError):
        function(image_3d, sigma=1)

    # extended_function should handle 3D by processing 2D slices
    result = extended_function(image_3d, sigma=1)
    assert result is not None
    assert result.shape == image_3d.shape
