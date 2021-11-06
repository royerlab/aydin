from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_butterworth import demo_butterworth
from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_butterworth():
    assert demo_butterworth(cropped_newyork(), display=False) >= 0.608 - 0.03


def test_butterworth_nd():
    check_nd(denoise_butterworth)
