"""Tests for the Gaussian mixture (GM) denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_gm import demo_gm
from aydin.it.classic_denoisers.gm import denoise_gm
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_gm():
    """Test GM denoiser on a 2D image achieves minimum SSIM."""
    assert demo_gm(cropped_newyork(crop_amount=384), display=False) >= 0.50


def test_gm_nd():
    """Test GM denoiser on 1D through 4D inputs."""
    check_nd(denoise_gm)
