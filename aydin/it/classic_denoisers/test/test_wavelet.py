from skimage.restoration import denoise_wavelet

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_wavelet import demo_wavelet
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_wavelet():
    assert demo_wavelet(cropped_newyork(), display=False) >= 0.587 - 0.01


def test_wavelet_nd():
    check_nd(denoise_wavelet)
