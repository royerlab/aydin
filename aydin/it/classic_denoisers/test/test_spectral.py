import pytest

from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_spectral import demo_spectral
from aydin.it.classic_denoisers.spectral import denoise_spectral
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


@pytest.mark.heavy
def test_spectral():
    assert demo_spectral(cropped_newyork(), display=False) >= 0.474 - 0.01


def test_spectral_nd():
    check_nd(denoise_spectral)
