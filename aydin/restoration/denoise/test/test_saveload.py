from os.path import join
import time

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise
from aydin.io.folders import get_temp_folder
from aydin.restoration.denoise.base import DenoiseRestorationBase


def saveload(denoiser, min_psnr=22, min_ssim=0.75):
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    denoiser.train(noisy, noisy)

    temp_file = join(get_temp_folder(), "test_it_saveload" + str(time.time()))
    print(temp_file)

    denoiser.save(temp_file)
    del denoiser

    loaded_denoiser = DenoiseRestorationBase()
    loaded_denoiser.load(temp_file)

    denoised = loaded_denoiser.denoise(noisy)

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
