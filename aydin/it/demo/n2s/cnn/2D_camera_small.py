# flake8: noqa
import time

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise
from aydin.it.cnn import ImageTranslatorCNN


def demo(image, max_epochs=4, image_width=200):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(image)
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    # CNN based Image translation:
    # input_dim only includes H, W, C; number of images is not included
    it = ImageTranslatorCNN(
        training_architecture='random',
        nb_unet_levels=3,
        batch_norm=None,  # 'instance',
        max_epochs=max_epochs,
    )

    start = time.time()
    # total_num_patches decides how many tiling batches to train.
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time: {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy, tile_size=image_width)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)
    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised_inf), name='denoised_inf')


camera_image = camera()
demo(camera_image)
