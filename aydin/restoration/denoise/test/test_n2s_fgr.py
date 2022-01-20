from pprint import pprint

import os
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import add_noise
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


def test_configure():

    implementations = Noise2SelfFGR().implementations
    pprint(implementations)

    configurable_arguments = Noise2SelfFGR().configurable_arguments
    pprint(configurable_arguments)

    implementations_description = Noise2SelfFGR().implementations_description
    pprint(implementations_description)


def test_run_n2s_fgr():
    # Prepare the noisy classic_denoisers camera image
    image = camera().astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    noisy_image = add_noise(image)

    # Call the Noise2Self restoration
    transforms = [
        {"class": RangeTransform, "kwargs": {}},
        {"class": PaddingTransform, "kwargs": {}},
    ]
    n2s = Noise2SelfFGR(variant="fgr-cb", it_transforms=transforms)
    n2s.train(noisy_image)
    denoised_image = n2s.denoise(noisy_image).clip(0, 1)

    # Check if denoised image satisfies some checks
    assert psnr(denoised_image, image) >= 20.0
    assert ssim(denoised_image, image) >= 0.7


def test_run_n2s_fgr_reusing_weights(tmpdir):
    # Prepare the noisy classic_denoisers camera image
    image = camera().astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    noisy_image_1 = add_noise(image)

    psnr_noisy = psnr(noisy_image_1, image)
    ssim_noisy = ssim(noisy_image_1, image)
    print("noisy 1", psnr_noisy, ssim_noisy)

    # Call the Noise2Self restoration
    transforms = [
        {"class": RangeTransform, "kwargs": {}},
        {"class": PaddingTransform, "kwargs": {}},
    ]
    n2s = Noise2SelfFGR(variant="fgr-cb", it_transforms=transforms)
    n2s.train(noisy_image_1)
    n2s.save_model(os.path.join(tmpdir, "denoise_model.zip"))
    n2s_loaded = Noise2SelfFGR(
        use_model=True, input_model_path=os.path.join(tmpdir, "denoise_model.zip")
    )
    denoised_image_loaded = n2s_loaded.denoise(noisy_image_1).clip(0, 1)

    psnr_denoised = psnr(denoised_image_loaded, image)
    ssim_denoised = ssim(denoised_image_loaded, image)
    print("denoised_trained", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 20 and ssim_denoised > 0.7

    noisy_image_2 = add_noise(image)
    denoised_image_2_loaded = n2s_loaded.denoise(noisy_image_1).clip(0, 1)

    psnr_denoised = psnr(denoised_image_2_loaded, image)
    ssim_denoised = ssim(denoised_image_2_loaded, image)
    print("denoised_trained", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 20 and ssim_denoised > 0.7
