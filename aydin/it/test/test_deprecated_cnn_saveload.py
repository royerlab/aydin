# flake8: noqa
import time
from os.path import join

import napari
import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise
from aydin.io.folders import get_temp_folder
from aydin.it.base import ImageTranslatorBase
from aydin.it.cnn import ImageTranslatorCNN


def test_saveload_cnn():
    max_epochs = 2
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        # training_architecture='random',
        model_architecture='jinet',
        nb_unet_levels=2,
        mask_size=3,
        batch_norm='instance',
        max_epochs=max_epochs,
    )
    it.train(noisy, noisy)

    # values before loading
    denoised = it.translate(noisy, tile_size=image_width)
    denoised = denoised.reshape(image.shape)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    temp_file = join(get_temp_folder(), "test_it_saveload_cnn" + str(time.time()))
    print(f"savepath: {temp_file}")
    it.save(temp_file)
    del it

    loaded_it = ImageTranslatorBase.load(temp_file)

    denoised1 = numpy.clip(loaded_it.translate(noisy), 0, 1)
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(noisy)
    #     viewer.add_image(denoised)
    #     viewer.add_image(denoised1)
    psnr_denoised1 = psnr(denoised1.squeeze(), image)
    ssim_denoised1 = ssim(denoised1.squeeze(), image)
    print("denoised", psnr_denoised1, ssim_denoised1)

    assert 0.99 < psnr_denoised / psnr_denoised1 < 1.01
    assert 0.99 < ssim_denoised / ssim_denoised1 < 1.01
