"""Demo of Noise2Self perceptron denoising on a 2D camera image.

Trains an ``ImageTranslatorFGR`` with ``PerceptronRegressor`` on the
camera image with synthetic noise, demonstrating basic self-supervised
denoising with a neural network regressor.
"""

# flake8: noqa
import time
from functools import partial

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.perceptron import PerceptronRegressor


def demo():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(camera().astype(np.float32, copy=False))
    noisy = add_noise(image)

    generator = StandardFeatureGenerator()  # max_level=4)
    regressor = PerceptronRegressor(max_epochs=1)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy)
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

    print("noisy", psnr(image, noisy), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))
    # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised, name='denoised')
    napari.run()


if __name__ == "__main__":
    demo()
