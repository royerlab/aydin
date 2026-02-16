"""Demonstrate stopping FGR training from another thread.

This demo shows how to programmatically stop an FGR training session from
a separate thread using a timer, which is useful for implementing training
time limits or user-triggered cancellation.
"""

# flake8: noqa
import time
from functools import partial

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastFeatureGenerator
from aydin.io.datasets import normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.util.log.log import Log


def demo():
    """Train FGR on noisy data and stop training after 20 seconds."""

    Log.set_log_max_depth(5)

    image = camera().astype(np.float32)
    image = normalise(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, rng=0)
    noisy = noisy.astype(np.float32)

    generator = FastFeatureGenerator(max_level=10)
    regressor = GBMRegressor()
    # regressor = NNRegressor()
    # regressor = CBRegressor()

    it = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, max_voxels_for_training=1e7
    )

    from threading import Timer

    def stop_training():
        """Signal the FGR trainer to stop from a background timer thread."""
        print("!!STOPPING TRAINING NOW FROM ANOTHER THREAD!!")
        it.stop_training()

    t = Timer(20.0, stop_training)
    t.start()

    start = time.time()
    denoised = it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised    :", psnr(image, denoised), ssim(denoised, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    viewer = napari.Viewer()
    viewer.add_image(n(image), name='image')
    viewer.add_image(n(noisy), name='noisy')
    viewer.add_image(n(denoised), name='denoised')
    viewer.add_image(n(denoised_inf), name='denoised_inf')
    napari.run()


if __name__ == "__main__":
    demo()
