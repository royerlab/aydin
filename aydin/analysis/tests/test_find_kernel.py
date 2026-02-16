"""Tests for blur kernel estimation."""

import numpy as np
from scipy.ndimage import uniform_filter

from aydin.analysis.find_kernel import compute_relative_blur_kernel


def test_kernel_shape():
    """Output kernel should have the requested size."""
    rng = np.random.RandomState(0)
    clean = rng.randn(64, 64).astype(np.float32)
    blurry = uniform_filter(clean, size=3)
    kernel = compute_relative_blur_kernel(clean, blurry, size=5)
    assert kernel.shape == (5, 5)


def test_kernel_sums_to_one():
    """Kernel should be normalized to sum to 1."""
    rng = np.random.RandomState(0)
    clean = rng.randn(64, 64).astype(np.float32)
    blurry = uniform_filter(clean, size=3)
    kernel = compute_relative_blur_kernel(clean, blurry, size=5)
    np.testing.assert_almost_equal(kernel.sum(), 1.0, decimal=5)


def test_identity_kernel():
    """When clean and blurry are the same, kernel should peak in the center."""
    rng = np.random.RandomState(0)
    image = rng.randn(64, 64).astype(np.float32) + 10  # offset to avoid division issues
    kernel = compute_relative_blur_kernel(image, image, size=3)
    # Center pixel should be the maximum
    center = kernel[1, 1]
    assert center == kernel.max()


def test_3d_kernel():
    """Should work for 3D images."""
    rng = np.random.RandomState(0)
    clean = rng.randn(32, 32, 32).astype(np.float32)
    blurry = uniform_filter(clean, size=3)
    kernel = compute_relative_blur_kernel(clean, blurry, size=3)
    assert kernel.shape == (3, 3, 3)
    np.testing.assert_almost_equal(kernel.sum(), 1.0, decimal=5)
