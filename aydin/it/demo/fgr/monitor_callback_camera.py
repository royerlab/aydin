# flake8: noqa
import time

import napari
import numpy
import numpy as np
from aydin.features.fast.fast_features import FastFeatureGenerator
from aydin.it.monitor import Monitor
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.lgbm import LGBMRegressor
from aydin.regression.nn import NNRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
    )


def demo(regressor):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = camera().astype(np.float32)
    image = n(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()

        size = 128
        monitoring_image = noisy[
            256 - size // 2 : 256 + size // 2, 256 - size // 2 : 256 + size // 2
        ]

        def callback(arg):
            # print(arg)
            iteration, val_loss, image = arg
            print(
                f"********CALLBACK******** --> Iteration: {iteration} metric: {val_loss}"
            )
            # print(f"images: {str(images)}")
            # print("image: ", image[0])
            if image[0] is not None:
                print(
                    f"********CALLBACK******** --> Image: shape={image[0].shape} dtype={image[0].dtype}"
                )
                viewer.add_image(
                    rescale_intensity(image[0], in_range='image', out_range=(0, 1)),
                    name=f'noisy{iteration}',
                )

        generator = FastFeatureGenerator()

        monitor = Monitor(
            monitoring_callbacks=[callback], monitoring_images=[monitoring_image]
        )

        it = ImageTranslatorFGR(
            feature_generator=generator, regressor=regressor, monitor=monitor
        )

        start = time.time()

        it.train(noisy, noisy)

        stop = time.time()
        print(f"Training: elapsed time:  {stop - start} ")

        denoised = it.translate(noisy)
        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))

        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(denoised), name='denoised')


demo(NNRegressor(max_epochs=10))
demo(LGBMRegressor(max_num_estimators=400))
