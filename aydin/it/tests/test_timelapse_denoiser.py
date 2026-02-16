"""Tests for TimelapseDenoiser."""

import numpy
import pytest

from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.timelapse_denoiser import TimelapseDenoiser


def test_timelapse_denoiser_init():
    """Test TimelapseDenoiser initialization with default parameters."""
    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(translator)

    assert denoiser.translator is translator
    assert denoiser.fine_temporal_window == 1
    assert denoiser.coarse_temporal_window == 7
    assert denoiser.use_median is True


def test_timelapse_denoiser_init_custom_params():
    """Test TimelapseDenoiser initialization with custom parameters."""
    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(
        translator,
        fine_temporal_window=2,
        coarse_temporal_window=5,
        use_median=False,
    )

    assert denoiser.fine_temporal_window == 2
    assert denoiser.coarse_temporal_window == 5
    assert denoiser.use_median is False


@pytest.mark.heavy
def test_timelapse_denoiser_denoise():
    """Test basic denoising functionality on a small synthetic timelapse."""
    # Create a small synthetic timelapse (5 time points, 32x32 images)
    numpy.random.seed(42)
    clean = numpy.random.rand(5, 32, 32).astype(numpy.float32) * 0.5 + 0.25
    noisy = clean + numpy.random.randn(5, 32, 32).astype(numpy.float32) * 0.1
    noisy = numpy.clip(noisy, 0, 1)

    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(
        translator,
        fine_temporal_window=1,
        coarse_temporal_window=2,
    )

    # Denoise only a single time point for speed
    denoised = denoiser.denoise(noisy, interval=(2, 3))

    assert denoised.shape == noisy.shape
    assert denoised.dtype == noisy.dtype


@pytest.mark.heavy
def test_timelapse_denoiser_denoise_output_array():
    """Test that denoise can use a pre-allocated output array."""
    numpy.random.seed(42)
    # Use larger images (64x64) to avoid edge cases in feature extraction
    noisy = numpy.random.rand(5, 64, 64).astype(numpy.float32)
    output = numpy.zeros_like(noisy)

    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(
        translator,
        fine_temporal_window=1,
        coarse_temporal_window=2,
    )

    # Denoise into pre-allocated array (single timepoint for speed)
    result = denoiser.denoise(noisy, denoised_image_array=output, interval=(2, 3))

    # Result should be the same object as output
    assert result is output


@pytest.mark.heavy
def test_timelapse_denoiser_denoise_creates_output():
    """Test that denoise creates output array when not provided."""
    numpy.random.seed(42)
    noisy = numpy.random.rand(5, 64, 64).astype(numpy.float32)

    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(
        translator,
        fine_temporal_window=1,
        coarse_temporal_window=2,
    )

    # Don't provide output array
    result = denoiser.denoise(noisy, interval=(2, 3))

    assert result is not None
    assert result.shape == noisy.shape


@pytest.mark.heavy
def test_timelapse_denoiser_use_median_false():
    """Test TimelapseDenoiser with use_median=False (uses mean instead)."""
    numpy.random.seed(42)
    noisy = numpy.random.rand(5, 64, 64).astype(numpy.float32)

    translator = ImageTranslatorFGR()
    denoiser = TimelapseDenoiser(
        translator,
        fine_temporal_window=1,
        coarse_temporal_window=2,
        use_median=False,  # Use mean instead
    )

    result = denoiser.denoise(noisy, interval=(2, 3))
    assert result.shape == noisy.shape
