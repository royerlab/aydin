# flake8: noqa
import time

import numpy
import numpy as np
from skimage.data import astronaut
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import rgbtest
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import Log


def demo(image):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True

    amplitude = 190
    noisy = image.astype(np.int16) + np.random.randint(
        -amplitude, amplitude, size=image.shape, dtype=np.int16
    )
    noisy = noisy.clip(0, 255).astype(np.uint8)

    # image = image.astype(np.float32)
    # noisy = noisy.astype(np.float32)

    generator = StandardFeatureGenerator(include_spatial_features=True)
    regressor = LGBMRegressor(learning_rate=0.01)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, channel_axes=(False, False, True))
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(noisy, channel_axes=(False, False, True))
    stop = time.time()
    print(f"Inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 255)
    noisy = numpy.clip(noisy, 0, 255)
    denoised = numpy.clip(denoised, 0, 255)

    print(
        "noisy",
        psnr(image, noisy, data_range=255),
        ssim(noisy, image, multichannel=True),
    )
    print(
        "denoised",
        psnr(image, denoised, data_range=255),
        ssim(denoised, image, multichannel=True),
    )
    # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image', rgb=True)
        viewer.add_image(noisy, name='noisy', rgb=True)
        viewer.add_image(denoised, name='denoised', rgb=True)

        cm = {0: 'r', 1: 'g', 2: 'b'}
        for i in range(3):
            viewer.add_image(image[..., i], name=f'image {cm[i]}')
            viewer.add_image(noisy[..., i], name=f'noisy {cm[i]}')
            viewer.add_image(denoised[..., i], name=f'denoised {cm[i]}')


if __name__ == "__main__":
    image = rgbtest()
    demo(image)
    image = astronaut()
    demo(image)
