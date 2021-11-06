from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_nlm import demo_nlm
from aydin.it.classic_denoisers.nlm import denoise_nlm
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_nlm():
    assert demo_nlm(cropped_newyork(), display=False) >= 0.626 - 0.01


def test_nlm_nd():
    check_nd(denoise_nlm)
