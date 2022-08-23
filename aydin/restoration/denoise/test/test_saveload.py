from os.path import join
import time

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin import Classic
from aydin.io.datasets import normalise, add_noise
from aydin.io.folders import get_temp_folder
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


transforms = [
    {"class": RangeTransform, "kwargs": {}},
    {"class": PaddingTransform, "kwargs": {}},
]


def test_saveload_classic_gaussian():
    saveload(
        Classic(variant="gaussian", it_transforms=transforms),
        min_psnr=19,
        min_ssim=0.61,
    )


def test_saveload_noise2selffgr():
    saveload(
        Noise2SelfFGR(variant="cb", it_transforms=transforms),
        min_psnr=20,
        min_ssim=0.78,
    )


def saveload(denoiser, min_psnr=22, min_ssim=0.75):
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    denoiser.train(noisy)

    temp_file = join(get_temp_folder(), "test_restoration_saveload" + str(time.time()))
    print(temp_file)

    denoiser.save(temp_file)
    loaded_denoiser = denoiser.__class__()

    del denoiser

    loaded_denoiser.load(temp_file + ".zip")

    denoised = loaded_denoiser.denoise(noisy)

    denoised = denoised.clip(0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    # Check if denoised image satisfies some checks
    assert psnr_denoised >= 20.0
    assert ssim_denoised >= 0.7

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
