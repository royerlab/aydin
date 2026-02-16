"""Demo script for 2D Lipschitz continuity denoising with auto-calibration."""

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

from aydin.io.datasets import (
    add_noise,
    characters,
    dots,
    lizard,
    newyork,
    normalise,
    pollen,
)
from aydin.it.classic_denoisers.lipschitz import calibrate_denoise_lipschitz
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


def demo_lipschitz(image, display=True):
    """Denoise a 2D image using auto-calibrated Lipschitz continuity constraints."""
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image, intensity=64, variance=0.001, sap=0.2)
    noisy = add_noise(noisy, intensity=128, variance=0.01, sap=0)

    start = time.time()
    function, parameters, memreq = calibrate_denoise_lipschitz(noisy)
    denoised = function(noisy, **parameters)
    stop = time.time()
    print(f"Denoising: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy, data_range=1.0)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised, data_range=1.0)
    print("           noisy   :", psnr_noisy, ssim_noisy)
    print("lipschitz denoised:", psnr_denoised, ssim_denoised)

    if display:
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
        plt.savefig(os.path.join(_DEMO_RESULTS, 'classic_lipschitz_2D.png'))

    return ssim_denoised


if __name__ == "__main__":
    demo_lipschitz(newyork())
    demo_lipschitz(characters())
    demo_lipschitz(pollen())
    demo_lipschitz(lizard())
    demo_lipschitz(dots())
    demo_lipschitz(camera())
