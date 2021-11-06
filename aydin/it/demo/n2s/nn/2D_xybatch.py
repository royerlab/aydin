# flake8: noqa
import time

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, add_noise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.nn import NNRegressor


def demo():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(camera().astype(np.float32, copy=False))
    noisy = add_noise(image)

    generator = StandardFeatureGenerator()  # max_level=4)
    regressor = NNRegressor(max_epochs=1)

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

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))
    # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')


demo()
