from scipy.ndimage import gaussian_filter

from aydin.io.datasets import cropped_newyork, dots, dmel
from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.it.classic_denoisers.demo.demo_2D_butterworth import demo_butterworth
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_butterworth():
    ssim_denoised, parameters = demo_butterworth(cropped_newyork(), display=False)
    assert ssim_denoised >= 0.608 - 0.035


def test_butterworth_anisotropy():
    image = dmel()

    image = gaussian_filter(image, sigma=[0.001, 4])

    ssim_denoised, parameters = demo_butterworth(image, display=True)

    print(parameters)
    cutoffs = parameters['freq_cutoff']
    assert cutoffs[0] > 2 * cutoffs[1]


def test_butterworth_nd():
    check_nd(denoise_butterworth)
