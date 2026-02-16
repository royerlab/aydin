"""Tests for the spectral denoiser."""

import pytest

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_spectral import demo_spectral
from aydin.it.classic_denoisers.spectral import denoise_spectral
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


@pytest.mark.heavy
def test_spectral():
    """Test spectral denoiser on a 2D image achieves minimum SSIM."""
    assert demo_spectral(cropped_newyork(), display=False) >= 0.47


def test_spectral_nd():
    """Test spectral denoiser on 1D through 4D inputs."""
    check_nd(denoise_spectral)
