from aydin.io.datasets import cropped_newyork

from aydin.it.classic_denoisers.demo.demo_2D_tv import demo_tv
from aydin.it.classic_denoisers.test.util_test_nd import check_nd
from aydin.it.classic_denoisers.tv import denoise_tv


def test_tv():
    assert demo_tv(cropped_newyork(), display=False) >= 0.627 - 0.03


def test_tv_nd():
    check_nd(denoise_tv)
