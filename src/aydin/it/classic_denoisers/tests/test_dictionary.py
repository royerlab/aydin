"""Tests for the dictionary-based denoisers (fixed and learned)."""

import pytest

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_dictionary_fixed import (
    demo_dictionary_fixed,
)
from aydin.it.classic_denoisers.demo.demo_2D_dictionary_learned import (
    demo_dictionary_learned,
)
from aydin.it.classic_denoisers.dictionary_fixed import denoise_dictionary_fixed
from aydin.it.classic_denoisers.dictionary_learned import denoise_dictionary_learned
from aydin.it.classic_denoisers.tests.util_test_nd import check_nd


def test_dictionary_learned():
    """Test learned dictionary denoiser on a 2D image achieves minimum SSIM."""
    assert demo_dictionary_learned(cropped_newyork(), display=False) >= 0.50


@pytest.mark.heavy
def test_dictionary_fixed():
    """Test fixed dictionary denoiser on a 2D image achieves minimum SSIM."""
    assert demo_dictionary_fixed(cropped_newyork(), display=False) >= 0.636 - 0.02


def test_dictionary_fixed_nd():
    """Test fixed dictionary denoiser on 1D through 4D inputs."""
    check_nd(denoise_dictionary_fixed)


def test_dictionary_learned_nd():
    """Test learned dictionary denoiser on 1D through 4D inputs."""
    check_nd(denoise_dictionary_learned)
