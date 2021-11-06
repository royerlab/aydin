# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_lipschitz import demo_lipschitz

from aydin.it.classic_denoisers.demo.demo_2D_nlm import demo_nlm
from aydin.it.classic_denoisers.nlm import denoise_nlm
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_lipschitz():
    assert demo_lipschitz(cropped_newyork(), display=False) >= 0.50 - 0.02


def test_lipschitz_nd():
    check_nd(denoise_nlm)
