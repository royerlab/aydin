"""Tests for the Block-Matching nD (BMnD) denoiser."""

import numpy
import pytest

from aydin.io.datasets import add_noise, cropped_newyork, normalise
from aydin.it.classic_denoisers.bmnd import calibrate_denoise_bmnd, denoise_bmnd
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_bmnd_basic():
    """Test BMND denoiser on 2D image."""
    numpy.random.seed(42)
    image = cropped_newyork(crop_amount=400)
    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    denoised = denoise_bmnd(noisy, patch_size=5, block_depth=16)

    assert denoised.shape == noisy.shape
    assert denoised.dtype == numpy.float32
    # Verify denoising occurred (output differs from input)
    assert not numpy.array_equal(denoised, noisy)


def test_bmnd_mode_median():
    """Test BMND with median aggregation mode."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_bmnd(noisy, patch_size=5, block_depth=8, mode='median')

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)


def test_bmnd_mode_mean():
    """Test BMND with mean aggregation mode."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_bmnd(noisy, patch_size=5, block_depth=8, mode='mean')

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)


def test_bmnd_invalid_mode():
    """Test BMND raises error for invalid mode."""
    numpy.random.seed(42)
    image = numpy.random.rand(32, 32).astype(numpy.float32)

    with pytest.raises(ValueError, match="Unsupported mode"):
        denoise_bmnd(image, patch_size=5, mode='invalid')


def test_bmnd_calibration():
    """Test auto-calibration returns valid parameters."""
    numpy.random.seed(42)
    image = numpy.random.rand(64, 64).astype(numpy.float32)

    func, params, mem = calibrate_denoise_bmnd(image)

    assert func == denoise_bmnd
    assert isinstance(params, dict)
    assert mem > 0


def test_bmnd_nd():
    """Test BMND denoiser works on 1D-4D inputs."""
    check_nd(denoise_bmnd)


@pytest.mark.heavy
def test_bmnd_3d():
    """Test BMND on 3D volume."""
    numpy.random.seed(42)
    image = numpy.random.rand(16, 32, 32).astype(numpy.float32)
    noisy = image + 0.1 * numpy.random.randn(*image.shape).astype(numpy.float32)

    denoised = denoise_bmnd(noisy, patch_size=3, block_depth=8)

    assert denoised.shape == noisy.shape
    assert not numpy.array_equal(denoised, noisy)
