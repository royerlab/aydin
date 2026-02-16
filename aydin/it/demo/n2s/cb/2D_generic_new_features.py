"""Demonstrate 2D Noise2Self denoising with CatBoost using alternative features.

This demo applies self-supervised FGR denoising with CatBoost using a
feature generator configured with sinusoidal and lowpass features instead
of the default feature set, reporting PSNR/SSIM metrics and saving
comparison plots.
"""

# flake8: noqa
import os
import time
from functools import partial

import numpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import (
    add_noise,
    camera,
    cropped_newyork,
    dots,
    newyork,
    normalise,
)
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log

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


def demo(image, name='image', do_add_noise=True):
    """Denoise a 2D image using FGR with CatBoost and alternative features.

    Parameters
    ----------
    image : numpy.ndarray
        Input 2D image.
    name : str, optional
        Name used for labeling the saved output plot, by default 'image'.
    do_add_noise : bool, optional
        Whether to add synthetic noise to the image, by default True.
    """

    Log.enable_output = True

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image) if do_add_noise else image

    generator = StandardFeatureGenerator(
        include_corner_features=False,
        include_scale_one=True,
        include_fine_features=False,
        include_spatial_features=False,
        include_median_features=False,
        include_dct_features=False,
        num_sinusoidal_features=4,
        include_random_conv_features=False,
        include_lowpass_features=True,
    )

    regressor = CBRegressor(patience=256, gpu=True, min_num_estimators=1024)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

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
    plt.savefig(os.path.join(_DEMO_RESULTS, f'n2s_cb_newfeatures_2D_{name}.png'))


if __name__ == "__main__":
    newyork_image = newyork()
    demo(newyork_image, "newyork")
    camera_image = camera()
    demo(camera_image, "camera")

    newyork_image = cropped_newyork()
    demo(newyork_image, "newyork_cropped")
    dots_image = dots()
    demo(dots_image, "dots")
