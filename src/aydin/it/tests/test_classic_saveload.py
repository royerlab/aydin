"""Tests for ImageDenoiserClassic save/load round-trip functionality."""

import time
from os.path import join

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.analysis.image_metrics import ssim
from aydin.io.datasets import add_noise, normalise
from aydin.io.folders import get_temp_folder
from aydin.it.base import ImageTranslatorBase
from aydin.it.classic import ImageDenoiserClassic


@pytest.mark.parametrize(
    "method, min_psnr, min_ssim",
    [
        ("gaussian", 20, 0.40),
        ("lipschitz", 12, 0.10),
    ],
)
def test_saveload_gaussian(method, min_psnr, min_ssim):
    """Test save/load round-trip for lightweight classic denoiser methods."""
    saveload(method, min_psnr, min_ssim)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "method, min_psnr, min_ssim",
    [
        ("bilateral", 16, 0.30),
        ("gm", 20, 0.65),
        ("pca", 20, 0.60),
        ("tv", 20, 0.73),
        ("wavelet", 17, 0.40),
    ],
)
def test_saveload_wavelet(method, min_psnr, min_ssim):
    """Test save/load round-trip for heavy classic denoiser methods."""
    saveload(method, min_psnr, min_ssim)


def saveload(method, min_psnr=22, min_ssim=0.75):
    """Train, save, load, and verify a classic denoiser preserves quality.

    Parameters
    ----------
    method : str
        Name of the classic denoising method.
    min_psnr : float, optional
        Minimum acceptable PSNR after denoising.
    min_ssim : float, optional
        Minimum acceptable SSIM after denoising.
    """
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    it = ImageDenoiserClassic(method=method)

    it.train(noisy, noisy)

    temp_file = join(get_temp_folder(), "test_it_saveload" + str(time.time()))
    print(temp_file)

    it.save(temp_file)
    del it

    loaded_it = ImageTranslatorBase.load(temp_file)

    denoised = loaded_it.translate(noisy)

    denoised = denoised.clip(0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the
    # image the lgbm regressor have been broken.
    # do not change the number below, but instead, fix the
    # problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim


def test_classic_denoiser_repr():
    """Test that __repr__ is well-formed."""
    it = ImageDenoiserClassic(method='butterworth')
    r = repr(it)
    assert r.startswith('<')
    assert r.endswith('>')
    assert 'butterworth' in r
