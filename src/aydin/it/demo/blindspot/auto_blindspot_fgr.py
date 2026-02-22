"""Demonstrate automatic blind-spot detection with FGR-based denoising.

This demo compares Feature Generation and Regression (FGR) denoising with
and without automatically detected extended blind spots on a synthetically
noised camera image with correlated noise. Reports PSNR/SSIM metrics and
saves comparison plots.
"""

# flake8: noqa
import os
from functools import partial

import numpy
import numpy as np
import scipy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

from aydin.features.fast.fast_features import FastFeatureGenerator

ssim = partial(structural_similarity, data_range=1.0)

from aydin.io.datasets import add_noise, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.lgbm import LGBMRegressor
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


def demo(image, name):
    """Compare FGR denoising with and without auto-detected blind spots.

    Parameters
    ----------
    image : numpy.ndarray
        Clean 2D reference image.
    name : str
        Name used for labeling the saved output plot.
    """

    Log.enable_output = True

    image = image[0:510, 0:510]
    image = normalise(image.astype(np.float32))

    noisy = add_noise(image)
    kernel = numpy.array([[0.25, 0.5, 0.25]])
    noisy = scipy.ndimage.convolve(noisy, kernel, mode='mirror')

    # brute-force feature-exclusion based extended blindspot:
    generator = FastFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_line_features=True,
        include_spatial_features=True,
    )
    regressor = LGBMRegressor(patience=20, max_num_estimators=2048)

    itnbs = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    itwbs = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, blind_spots='auto'
    )

    itnbs.train(noisy)
    denoised_nbs = itnbs.translate(noisy)

    itwbs.train(noisy)
    denoised_wbs = itwbs.translate(noisy)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised_nbs = numpy.clip(denoised_nbs, 0, 1)
    denoised_wbs = numpy.clip(denoised_wbs, 0, 1)

    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised_nbs = psnr(image, denoised_nbs)
    ssim_denoised_nbs = ssim(image, denoised_nbs)
    psnr_denoised_wbs = psnr(image, denoised_wbs)
    ssim_denoised_wbs = ssim(image, denoised_wbs)

    print("noisy      :", psnr_noisy, ssim_noisy)
    print("denoised no extended blind-spot  :", psnr_denoised_nbs, ssim_denoised_nbs)
    print("denoised with extended blind-spot:", psnr_denoised_wbs, ssim_denoised_wbs)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised_nbs, name='denoised_nbs')
    viewer.add_image(denoised_wbs, name='denoised_wbs')
    napari.run()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(normalise(noisy), cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy \nPSNR: {psnr_noisy:.3f}, SSIM: {ssim_noisy:.3f}')
    plt.subplot(1, 4, 2)
    plt.imshow(normalise(denoised_nbs), cmap='gray')
    plt.axis('off')
    plt.title(
        f'Denoised (no BS) \nPSNR: {psnr_denoised_nbs:.3f}, SSIM: {ssim_denoised_nbs:.3f}'
    )
    plt.subplot(1, 4, 3)
    plt.imshow(normalise(denoised_wbs), cmap='gray')
    plt.axis('off')
    plt.title(
        f'Denoised (auto BS) \nPSNR: {psnr_denoised_wbs:.3f}, SSIM: {ssim_denoised_wbs:.3f}'
    )
    plt.subplot(1, 4, 4)
    plt.imshow(normalise(image), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1)
    os.makedirs(_DEMO_RESULTS, exist_ok=True)
    plt.savefig(os.path.join(_DEMO_RESULTS, f'auto_blindspot_fgr_{name}.png'))


if __name__ == "__main__":
    camera_image = camera()
    demo(camera_image, "camera")
