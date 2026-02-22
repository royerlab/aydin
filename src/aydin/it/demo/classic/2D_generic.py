"""Demonstrate 2D classic (spectral) denoising on various standard test images.

This demo applies the classic spectral denoiser via ``ImageDenoiserClassic``
to several 2D images with synthetic noise, reporting PSNR/SSIM metrics and
saving comparison plots.
"""

# flake8: noqa
import os
import time
from functools import partial

import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import (
    add_noise,
    characters,
    dots,
    lizard,
    newyork,
    normalise,
    pollen,
)
from aydin.it.classic import ImageDenoiserClassic
from aydin.it.classic_denoisers.spectral import (
    calibrate_denoise_spectral,
    denoise_spectral,
)
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log

_DEMO_RESULTS = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        '..',
        '..',
        'demo_results',
    )
)


def demo(image, name, do_add_noise=True):
    """Denoise a 2D image using the classic spectral denoiser.

    Parameters
    ----------
    image : numpy.ndarray
        Input 2D image.
    name : str
        Name used for labeling the saved output plot.
    do_add_noise : bool, optional
        Whether to add synthetic noise to the image, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image) if do_add_noise else image

    it = ImageDenoiserClassic(method='spectral')

    it.transforms_list.append(RangeTransform())
    it.transforms_list.append(PaddingTransform())

    print("training starts")

    start = time.time()
    it.train(noisy, noisy)
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
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("noisy   :", psnr_noisy, ssim_noisy)
    print("denoised:", psnr_denoised, ssim_denoised)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised, name='denoised')
    napari.run()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(normalise(noisy), cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy \nPSNR: {psnr_noisy:.3f}, SSIM: {ssim_noisy:.3f}')
    plt.subplot(1, 3, 2)
    plt.imshow(normalise(denoised), cmap='gray')
    plt.axis('off')
    plt.title(f'Denoised \nPSNR: {psnr_denoised:.3f}, SSIM: {ssim_denoised:.3f}')
    plt.subplot(1, 3, 3)
    plt.imshow(normalise(image), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1)
    os.makedirs(_DEMO_RESULTS, exist_ok=True)
    plt.savefig(os.path.join(_DEMO_RESULTS, f'classic_spectral_2D_{name}.png'))


if __name__ == "__main__":
    newyork_image = newyork()
    demo(newyork_image, "newyork")
    characters_image = characters()
    demo(characters_image, "characters")
    lizard_image = lizard()
    demo(lizard_image, "lizard")
    camera_image = camera()
    demo(camera_image, "camera")
    pollen_image = pollen()
    demo(pollen_image, "pollen")
    dots_image = dots()
    demo(dots_image, "dots")
