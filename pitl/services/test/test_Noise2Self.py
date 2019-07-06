import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from ..Noise2Self import Noise2Self


def test_run():
    # Prepare the noisy classical camera image
    image = camera().astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    intensity = 5
    np.random.seed(0)
    noisy_image = np.random.poisson(image * intensity) / intensity
    noisy_image = random_noise(noisy_image, mode='gaussian', var=0.01, seed=0)
    noisy_image = noisy_image.astype(np.float32)

    # Call the Noise2Self service
    n2s = Noise2Self()
    denoised_image = n2s.run(noisy_image)

    # Check if denoised image satisfies some checks
    assert psnr(denoised_image, image) >= 20.0
    assert ssim(denoised_image, image) >= 0.8
