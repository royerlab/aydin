"""Demo of Noise2Self LGBM denoising on generic 2D images.

Trains an ``ImageTranslatorFGR`` with ``LGBMRegressor`` and
``RangeTransform`` on several standard test images (New York, camera,
lizard, pollen, dots) and compares results against NLM and median
baselines.
"""

# flake8: noqa
import os
import time
from functools import partial

import numpy
import numpy as np
import skimage
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, dots, lizard, newyork, normalise, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.range import RangeTransform
from aydin.regression.lgbm import LGBMRegressor
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


def demo(image, name):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    Log.enable_output = True

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    median1 = skimage.filters.median(noisy, disk(1))
    median2 = skimage.filters.median(noisy, disk(2))
    median5 = skimage.filters.median(noisy, disk(5))

    nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = LGBMRegressor(
        patience=128,
        # loss='poisson',
        compute_training_loss=True,
    )

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.transforms_list.append(RangeTransform())

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
    plt.savefig(os.path.join(_DEMO_RESULTS, f'n2s_lgbm_2D_{name}.png'))

    plt.clf()
    plt.plot(regressor.loss_history[0]['training'], 'r')
    plt.plot(regressor.loss_history[0]['validation'], 'b')
    plt.legend(['training', 'validation'])
    plt.show()

    import napari

    viewer = napari.Viewer()
    viewer.add_image(normalise(image), name='image')
    viewer.add_image(normalise(noisy), name='noisy')
    viewer.add_image(normalise(nlm), name='nlm')
    viewer.add_image(normalise(median1), name='median1')
    viewer.add_image(normalise(median2), name='median2')
    viewer.add_image(normalise(median5), name='median5')
    viewer.add_image(normalise(denoised), name='denoised')
    napari.run()


if __name__ == "__main__":
    demo(newyork(), "newyork")
    demo(camera(), "camera")
    demo(lizard(), "lizard")
    demo(pollen(), "pollen")
    demo(dots(), "dots")
