"""Tests for the Lipschitz denoiser."""

# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_lipschitz import demo_lipschitz
from aydin.it.classic_denoisers.lipschitz import denoise_lipschitz
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_lipschitz():
    """Test Lipschitz denoiser on a 2D image achieves minimum SSIM."""
    assert demo_lipschitz(cropped_newyork(crop_amount=384), display=False) >= 0.43


def test_lipschitz_nd():
    """Test Lipschitz denoiser on 1D through 4D inputs."""
    check_nd(denoise_lipschitz)
