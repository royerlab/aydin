import pytest

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.bilateral import denoise_bilateral
from aydin.it.classic_denoisers.demo.demo_2D_bilateral import demo_bilateral
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


@pytest.mark.unstable
def test_bilateral():
    # Note: bilateral filter in scikit image may be unstable
    assert demo_bilateral(cropped_newyork(crop_amount=384), display=False) >= 0.38


def test_bilateral_nd():
    check_nd(denoise_bilateral)
