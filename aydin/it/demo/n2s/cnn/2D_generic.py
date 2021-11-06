# flake8: noqa
import time

import numpy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise, newyork
from aydin.it.cnn import ImageTranslatorCNN


def demo(image, max_epochs=10):
    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    # CNN based Image translation:
    # input_dim only includes H, W, C; number of images is not included
    it = ImageTranslatorCNN(
        training_architecture='checkran',  # 'checkran',  #
        model_architecture='jinet',  # 'jinet',  #
        nb_unet_levels=2,
        batch_norm='instance',  # None,  #
        activation='ReLU',
        # patch_size=512,
        mask_size=3,
        # total_num_patches=400,
        # batch_size=40,
        max_epochs=max_epochs,
        learn_rate=0.01,
        # reduce_lr_factor=0.1,  # 0.3 for masking methods, 0.1 for jinet
        # patience=10,
    )
    it.verbose = 1

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy, tile_size=512)
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


# camera_image = camera()
# demo(camera_image)
# lizard_image = lizard()
# demo(lizard_image)
# pollen_image = pollen()
# demo(pollen_image)
newyork_image = newyork()
demo(newyork_image)
# characters_image = characters()
# demo(characters_image)
# fibsem_image = fibsem()
# demo(fibsem_image)
