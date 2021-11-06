# flake8: noqa
import time

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

from aydin.it.cnn import ImageTranslatorCNN


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    """
    Demo for how to stop training from an other thread.
    """

    image = camera().astype(np.float32)
    image = n(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=-1), axis=0)
    it = ImageTranslatorCNN()

    from threading import Timer

    def stop_training():
        print("!!STOPPING TRAINING NOW FROM ANOTHER THREAD!!")
        it.stop_training()

    t = Timer(30.0, stop_training)
    t.start()

    start = time.time()
    it.train(
        noisy,
        noisy,
        # batch_dims=batch_dims,
    )
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    denoised = rescale_intensity(
        denoised_inf.squeeze(), in_range='image', out_range=(0, 1)
    )

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.squeeze(), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    denoised_inf = numpy.clip(denoised_inf.squeeze(), 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(denoised_inf), name='denoised_inf')


demo()
