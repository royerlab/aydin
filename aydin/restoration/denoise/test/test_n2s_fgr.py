from pprint import pprint

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
