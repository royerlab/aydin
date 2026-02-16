"""Tests for the bilateral denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.bilateral import denoise_bilateral
from aydin.it.classic_denoisers.demo.demo_2D_bilateral import demo_bilateral
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_bilateral():
    """Test bilateral denoiser on a 2D image achieves minimum SSIM."""
    assert demo_bilateral(cropped_newyork(crop_amount=384), display=False) >= 0.38


def test_bilateral_nd():
    """Test bilateral denoiser on 1D through 4D inputs."""
    check_nd(denoise_bilateral)
