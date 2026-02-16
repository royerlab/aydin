"""Demo of Noise2Self CNN denoising on a small crop of the camera image.

Trains an ``ImageTranslatorCNNTorch`` on a cropped, noisy version of the
scikit-image camera image and reports PSNR/SSIM metrics.
"""

# flake8: noqa
import os
import time
from functools import partial

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.io.datasets import add_noise, normalise
from aydin.it.cnn_torch import ImageTranslatorCNNTorch

_DEMO_RESULTS = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        '..',
        '..',
        '..',
        'demo_results',
    )
)


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
    it = ImageTranslatorCNNTorch(
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
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(noisy, image)
    psnr_denoised = psnr(image, denoised_inf)
    ssim_denoised = ssim(denoised_inf, image)
    print("noisy       :", psnr_noisy, ssim_noisy)
    print("denoised_inf:", psnr_denoised, ssim_denoised)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(normalise(image), name='image')
    viewer.add_image(normalise(noisy), name='noisy')
    viewer.add_image(normalise(denoised_inf), name='denoised_inf')
    napari.run()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(normalise(noisy), cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy \nPSNR: {psnr_noisy:.3f}, SSIM: {ssim_noisy:.3f}')
    plt.subplot(1, 3, 2)
    plt.imshow(normalise(denoised_inf), cmap='gray')
    plt.axis('off')
    plt.title(f'Denoised \nPSNR: {psnr_denoised:.3f}, SSIM: {ssim_denoised:.3f}')
    plt.subplot(1, 3, 3)
    plt.imshow(normalise(image), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1)
    os.makedirs(_DEMO_RESULTS, exist_ok=True)
    plt.savefig(os.path.join(_DEMO_RESULTS, 'n2s_cnn_2D_camera_small.png'))


if __name__ == "__main__":
    camera_image = camera()
    demo(camera_image)
