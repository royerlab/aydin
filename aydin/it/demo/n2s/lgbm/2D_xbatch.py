# flake8: noqa
import time

import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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

    image = camera().astype(np.float32)
    image = normalise(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    generator = StandardFeatureGenerator(max_level=10)
    regressor = LGBMRegressor()

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    batch_dims = (True, False)

    start = time.time()
    it.train(noisy, noisy, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(noisy, batch_axes=batch_dims)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))
    # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo()
