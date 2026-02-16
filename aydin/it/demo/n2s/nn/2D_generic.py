"""Demo of Noise2Self denoising with a perceptron regressor on 2D images.

Trains an ``ImageTranslatorFGR`` with ``PerceptronRegressor`` on
multiple standard test images corrupted by synthetic noise and
evaluates PSNR/SSIM.
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
from aydin.io.datasets import add_noise, characters, lizard, newyork, normalise, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.perceptron import PerceptronRegressor

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
    """Run Noise2Self perceptron denoising on a 2D image with synthetic noise.

    Parameters
    ----------
    image : numpy.ndarray
        Clean 2D reference image.
    name : str
        Label used for saved output files.
    """
    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)
    # noisy=image

    start_time = time.time()

    generator = StandardFeatureGenerator()
    regressor = PerceptronRegressor()

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.train(noisy, noisy)

    elapsedtime = time.time() - start_time
    print(f"time elapsed: {elapsedtime} s")

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

    plt.figure(figsize=(2.7 * 5, 5), dpi=300)
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
    plt.savefig(os.path.join(_DEMO_RESULTS, f'n2s_nn_2D_{name}.png'))

    plt.clf()
    plt.plot(regressor.loss_history[0]['training'], 'r')
    plt.plot(regressor.loss_history[0]['validation'], 'b')
    plt.legend(['training', 'validation'])
    plt.show()

    import napari

    viewer = napari.Viewer()
    viewer.add_image(normalise(image), name='image')
    viewer.add_image(normalise(noisy), name='noisy')
    viewer.add_image(normalise(denoised), name='denoised')
    napari.run()


if __name__ == "__main__":
    camera_image = camera()
    demo(camera_image, "camera")
    lizard_image = lizard()
    demo(lizard_image, "lizard")
    pollen_image = pollen()
    demo(pollen_image, "pollen")
    newyork_image = newyork()
    demo(newyork_image, "newyork")
    characters_image = characters()
    demo(characters_image, "characters")
