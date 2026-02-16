"""Demo script for 2D wavelet-based denoising with auto-calibration."""

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
from aydin.it.classic_denoisers.wavelet import calibrate_denoise_wavelet
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


def demo_wavelet(image, display=True):
    """Denoise a 2D image using auto-calibrated wavelet thresholding."""
    Log.enable_output = True
    Log.set_log_max_depth(6)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    start = time.time()
    function, parameters, memreq = calibrate_denoise_wavelet(
        noisy,
        all_wavelets=True,
        # wavelet_name_filter='sym'
    )
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
    print("        noisy   :", psnr_noisy, ssim_noisy)
    print("wavelet denoised:", psnr_denoised, ssim_denoised)

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
        plt.savefig(os.path.join(_DEMO_RESULTS, 'classic_wavelet_2D.png'))

    return ssim_denoised


if __name__ == "__main__":

    dots_image = dots()
    demo_wavelet(dots_image)
    camera_image = camera()
    demo_wavelet(camera_image)
    newyork_image = newyork()
    demo_wavelet(newyork_image)
    characters_image = characters()
    demo_wavelet(characters_image)
    pollen_image = pollen()
    demo_wavelet(pollen_image)
    lizard_image = lizard()
    demo_wavelet(lizard_image)


##['bior1.1',
# 'bior1.3',
# 'bior1.5',
# 'bior2.2',
# 'bior2.4',
# 'bior2.6',
# 'bior2.8',
# 'bior3.1',
# 'bior3.3',
# 'bior3.5',
# 'bior3.7',
# 'bior3.9',
# 'bior4.4',
# 'bior5.5',
# 'bior6.8',
# 'cgau1',
# 'cgau2',
# 'cgau3',
# 'cgau4',
# 'cgau5',
# 'cgau6',
# 'cgau7',
# 'cgau8',
# 'cmor',
# 'coif1',
# 'coif2',
# 'coif3',
# 'coif4',
# 'coif5',
# 'coif6',
# 'coif7',
# 'coif8',
# 'coif9',
# 'coif10',
# 'coif11',
# 'coif12',
# 'coif13',
# 'coif14',
# 'coif15',
# 'coif16',
# 'coif17',
# 'db1',
# 'db2',
# 'db3',
# 'db4',
# 'db5',
# 'db6',
# 'db7',
# 'db8',
# 'db9',
# 'db10',
# 'db11',
# 'db12',
# 'db13',
# 'db14',
# 'db15',
# 'db16',
# 'db17',
# 'db18',
# 'db19',
# 'db20',
# 'db21',
# 'db22',
# 'db23',
# 'db24',
# 'db25',
# 'db26',
# 'db27',
# 'db28',
# 'db29',
# 'db30',
# 'db31',
# 'db32',
# 'db33',
# 'db34',
# 'db35',
# 'db36',
# 'db37',
# 'db38',
# 'dmey',
# 'fbsp',
# 'gaus1',
# 'gaus2',
# 'gaus3',
# 'gaus4',
# 'gaus5',
# 'gaus6',
# 'gaus7',
# 'gaus8',
# 'haar',
# 'mexh',
# 'morl',
# 'rbio1.1',
# 'rbio1.3',
# 'rbio1.5',
# 'rbio2.2',
# 'rbio2.4',
# 'rbio2.6',
# 'rbio2.8',
# 'rbio3.1',
# 'rbio3.3',
# 'rbio3.5',
# 'rbio3.7',
# 'rbio3.9',
# 'rbio4.4',
# 'rbio5.5',
# 'rbio6.8',
# 'shan',
# 'sym2',
# 'sym3',
# 'sym4',
# 'sym5',
# 'sym6',
# 'sym7',
# 'sym8',
# 'sym9',
# 'sym10',
# 'sym11',
# 'sym12',
# 'sym13',
# 'sym14',
# 'sym15',
# 'sym16',
# 'sym17',
# 'sym18',
# 'sym19',
# 'sym20']
