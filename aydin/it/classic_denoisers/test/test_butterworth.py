# flake8: noqa
from scipy.ndimage import gaussian_filter

from aydin.io.datasets import cropped_newyork, newyork
from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.it.classic_denoisers.demo.demo_2D_butterworth import demo_butterworth
from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_butterworth():
    ssim_denoised, parameters = demo_butterworth(
        cropped_newyork(crop_amount=384), display=False
    )
    assert ssim_denoised >= 0.54


def test_butterworth_anisotropy():
    # Use newyork image (always available) instead of dmel (requires download)
    image = newyork()[100:400, 100:400]
    # Apply anisotropic blur: heavy blur in one direction (sigma=7),
    # light blur in the other (sigma=0.5)
    image = gaussian_filter(image, sigma=[0.5, 7])

    ssim_denoised, parameters = demo_butterworth(image, display=False)

    print(parameters)
    cutoffs = parameters['freq_cutoff']
    # Anisotropic filtering should result in different cutoffs per axis
    assert cutoffs[0] > cutoffs[1]


def test_butterworth_nd():
    check_nd(denoise_butterworth)
