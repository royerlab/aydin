from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_gm import demo_gm
from aydin.it.classic_denoisers.gm import denoise_gm
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_gm():
    assert demo_gm(cropped_newyork(), display=False) >= 0.610 - 0.03


def test_gm_nd():
    check_nd(denoise_gm)
