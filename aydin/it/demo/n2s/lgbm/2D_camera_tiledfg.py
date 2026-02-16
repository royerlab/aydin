"""Demo of Noise2Self LGBM denoising with tiled feature generation.

Trains an ``ImageTranslatorFGR`` with ``LGBMRegressor`` using a low
``max_memory_usage_ratio`` to force tiled feature generation, and
compares the denoised result against NLM and median filters.
"""

# flake8: noqa
import time
from functools import partial

import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)
from skimage.util import random_noise

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import Log


def demo():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True

    image = camera().astype(np.float32)  # newyork()[256:-256, 256:-256]
    image = normalise(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, rng=0)
    noisy = noisy.astype(np.float32)

    generator = StandardFeatureGenerator(max_level=10)
    regressor = LGBMRegressor()

    it = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, max_memory_usage_ratio=0.0001
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised    :", psnr(image, denoised), ssim(denoised, image))

    import napari

    viewer = napari.Viewer()
    viewer.add_image(normalise(image), name='image')
    viewer.add_image(normalise(noisy), name='noisy')
    viewer.add_image(normalise(denoised), name='denoised')
    napari.run()


if __name__ == "__main__":
    demo()
