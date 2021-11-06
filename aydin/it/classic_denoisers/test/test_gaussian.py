from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_gaussian import demo_gaussian
from aydin.it.classic_denoisers.gaussian import denoise_gaussian
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_gaussian():
    assert demo_gaussian(cropped_newyork(), display=False) >= 0.600 - 0.02


def test_gaussian_nd():
    check_nd(denoise_gaussian)
