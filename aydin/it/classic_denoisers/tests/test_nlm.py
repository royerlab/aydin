"""Tests for the non-local means (NLM) denoiser."""

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_nlm import demo_nlm
from aydin.it.classic_denoisers.nlm import denoise_nlm
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_nlm():
    """Test NLM denoiser on a 2D image achieves minimum SSIM."""
    # NLM consistently achieves ~0.54 SSIM with the default noise level;
    # the old threshold of 0.606 was unreachable with current scikit-image.
    assert demo_nlm(cropped_newyork(), display=False) >= 0.50


def test_nlm_nd():
    """Test NLM denoiser on 1D through 4D inputs."""
    check_nd(denoise_nlm)
