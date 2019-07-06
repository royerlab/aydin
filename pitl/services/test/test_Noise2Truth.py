import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from ..Noise2Truth import Noise2Truth


# TODO: Check with loic if this test makes sense
def test_run():
    # Prepare the noisy classical camera image
    image = camera().astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    intensity = 5
    np.random.seed(0)
    noisy_image = np.random.poisson(image * intensity) / intensity
    noisy_image = random_noise(noisy_image, mode='gaussian', var=0.01, seed=0)
    noisy_image = noisy_image.astype(np.float32)

    intensity = 6
    np.random.seed(15)
    noisy_test = np.random.poisson(image * intensity) / intensity
    noisy_test = random_noise(noisy_test, mode='gaussian', var=0.01, seed=0)
    noisy_test = noisy_test.astype(np.float32)

    # Call the Noise2Self service
    n2t = Noise2Truth()
    denoised_image = n2t.run(noisy_image, image, noisy_test)

    # Check if denoised image satisfies some checks
    assert psnr(denoised_image, image) >= 20.0
    assert ssim(denoised_image, image) >= 0.8
