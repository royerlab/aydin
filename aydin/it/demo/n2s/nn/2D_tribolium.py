# flake8: noqa
import time
from os.path import join

import napari
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tifffile import imread

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_zipped, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.nn import NNRegressor


def demo():
    """
    Demo for supervised denoising using CARE's tribolium example as a montage.

    """

    image = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_gt_montage.tif'
        )
    ).astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    noisy = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_montage.tif'
        )
    ).astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    image_test = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_gt_montage.tif')
    ).astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1))

    noisy_test = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_montage.tif'
        )
    ).astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1))

    generator = StandardFeatureGenerator()

    regressor = NNRegressor(max_epochs=5)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(noisy_test)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')

        viewer.add_image(normalise(denoised), name='denoised')
        viewer.add_image(normalise(image_test), name='image_test')
        viewer.add_image(normalise(noisy_test), name='noisy_test')


demo()
