"""Tests for the PCA-based patch denoiser."""

import numpy
import pytest

from aydin.io.datasets import add_noise, cropped_newyork, normalise
from aydin.it.classic_denoisers.pca import calibrate_denoise_pca, denoise_pca
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_pca_basic():
    """Test PCA denoiser on 2D image."""
    numpy.random.seed(42)
    image = cropped_newyork(crop_amount=400)
    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    denoised = denoise_pca(noisy, patch_size=5, threshold=0.5)

    assert denoised.shape == noisy.shape
    assert denoised.dtype == numpy.float32
    # Verify denoising occurred (output differs from input)
    assert not numpy.array_equal(denoised, noisy)


def test_pca_threshold_low():
    """Test PCA with low threshold (fewer components, more aggressive denoising)."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_pca(noisy, patch_size=5, threshold=0.1)

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)


def test_pca_threshold_high():
    """Test PCA with high threshold (more components, less aggressive denoising)."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_pca(noisy, patch_size=5, threshold=0.9)

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)


def test_pca_calibration():
    """Test auto-calibration returns valid parameters."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)

    func, params, mem = calibrate_denoise_pca(image, max_num_evaluations=3)

    assert func == denoise_pca
    assert isinstance(params, dict)
    assert 'threshold' in params
    assert 0 <= params['threshold'] <= 1
    assert mem > 0


def test_pca_nd():
    """Test PCA denoiser works on 1D-4D inputs."""
    check_nd(denoise_pca)


@pytest.mark.heavy
def test_pca_3d():
    """Test PCA on 3D volume."""
    numpy.random.seed(42)
    image = numpy.random.rand(16, 32, 32).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_pca(noisy, patch_size=3, threshold=0.5)

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)
