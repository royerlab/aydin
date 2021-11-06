from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.bilateral import denoise_bilateral

from aydin.it.classic_denoisers.demo.demo_2D_bilateral import demo_bilateral
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def est_bilateral():
    # TODO: bilateral filter in scikit image seems broken
    assert demo_bilateral(cropped_newyork(), display=False) >= 0.40 - 0.1


def test_bilateral_nd():
    check_nd(denoise_bilateral)
