from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_harmonic import demo_harmonic
from aydin.it.classic_denoisers.harmonic import denoise_harmonic
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_harmonic():
    assert demo_harmonic(cropped_newyork(), display=False) >= 0.621 - 0.03


def test_harmonic_nd():
    check_nd(denoise_harmonic)
