"""Tests for the Gaussian denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_gaussian import demo_gaussian
from aydin.it.classic_denoisers.gaussian import denoise_gaussian
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_gaussian():
    """Test Gaussian denoiser on a 2D image achieves minimum SSIM."""
    assert demo_gaussian(cropped_newyork(crop_amount=384), display=False) >= 0.50


def test_gaussian_nd():
    """Test Gaussian denoiser on 1D through 4D inputs."""
    check_nd(denoise_gaussian)
