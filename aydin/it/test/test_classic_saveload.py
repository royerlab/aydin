import time
from os.path import join
import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise
from aydin.io.folders import get_temp_folder
from aydin.it.base import ImageTranslatorBase
from aydin.it.classic import ImageDenoiserClassic


@pytest.mark.heavy
def test_saveload_bilateral():
    saveload("bilateral", min_psnr=16, min_ssim=0.30)


def test_saveload_gaussian():
    saveload("gaussian", min_psnr=20, min_ssim=0.71)


@pytest.mark.heavy
def test_saveload_gm():
    saveload("gm", min_psnr=20, min_ssim=0.65)


def test_saveload_lipschitz():
    saveload("lipschitz", min_psnr=12, min_ssim=0.16)


# @pytest.mark.heavy
def test_saveload_pca():
    saveload("pca", min_psnr=20, min_ssim=0.60)


@pytest.mark.heavy
def test_saveload_tv():
    saveload("tv", min_psnr=20, min_ssim=0.73)


@pytest.mark.heavy
def test_saveload_wavelet():
    saveload("wavelet", min_psnr=17, min_ssim=0.40)


def saveload(method, min_psnr=22, min_ssim=0.75):
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

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
