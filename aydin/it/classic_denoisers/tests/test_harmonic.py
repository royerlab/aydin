"""Tests for the harmonic denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_harmonic import demo_harmonic
from aydin.it.classic_denoisers.harmonic import denoise_harmonic
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_harmonic():
    """Test harmonic denoiser on a 2D image achieves minimum SSIM."""
    assert demo_harmonic(cropped_newyork(), display=False) >= 0.50


def test_harmonic_nd():
    """Test harmonic denoiser on 1D through 4D inputs."""
    check_nd(denoise_harmonic)
