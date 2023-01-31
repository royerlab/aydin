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


@pytest.mark.parametrize("method, min_psnr, min_ssim", [
    ("gaussian", 20, 0.61),
    ("lipschitz", 12, 0.16),
])
def test_saveload_gaussian(method, min_psnr, min_ssim):
    saveload(method, min_psnr, min_ssim)


@pytest.mark.heavy
@pytest.mark.parametrize("method, min_psnr, min_ssim", [
    ("bilateral", 16, 0.30),
    ("gm", 20, 0.65),
    ("pca", 20, 0.60),
    ("tv", 20, 0.73),
    ("wavelet", 17, 0.40),
])
def test_saveload_wavelet(method, min_psnr, min_ssim):
    saveload(method, min_psnr, min_ssim)


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
