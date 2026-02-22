"""Tests for the total variation (TV) denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_tv import demo_tv
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd
from aydin.it.classic_denoisers.tv import denoise_tv


def test_tv():
    """Test TV denoiser on a 2D image achieves minimum SSIM."""
    assert demo_tv(cropped_newyork(), display=False) >= 0.50


def test_tv_nd():
    """Test TV denoiser on 1D through 4D inputs."""
    check_nd(denoise_tv)
