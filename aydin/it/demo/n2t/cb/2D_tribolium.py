# flake8: noqa
import time
from os.path import join

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tifffile import imread

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_zipped, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def demo():
    """
    Demo for supervised denoising using CARE's tribolium example as a montage.

    """

    image = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_gt_montage.tif'
        )
    ).astype(np.float32)
    image = normalise(image)

    noisy = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_montage.tif'
        )
    ).astype(np.float32)
    noisy = normalise(noisy)

    image_test = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_gt_montage.tif')
    ).astype(np.float32)
    image_test = normalise(image_test)

    noisy_test = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_montage.tif'
        )
    ).astype(np.float32)
    noisy_test = normalise(noisy_test)

    scales = [1, 3, 7, 15, 31]
    widths = [3, 3, 3, 3, 3]

    generator = StandardFeatureGenerator(kernel_widths=widths, kernel_scales=scales)

    regressor = CBRegressor()

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference train: elapsed time:  {stop - start} ")

    start = time.time()
    denoised_test = it.translate(noisy_test)
    stop = time.time()
    print(f"inference test: elapsed time:  {stop - start} ")

    denoised = denoised.clip(0, 1)
    denoised_test = denoised_test.clip(0, 1)

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))
    print(
        "denoised_test",
        psnr(denoised_test, image_test),
        ssim(denoised_test, image_test),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )

        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )
        viewer.add_image(
            rescale_intensity(image_test, in_range='image', out_range=(0, 1)),
            name='image_test',
        )
        viewer.add_image(
            rescale_intensity(noisy_test, in_range='image', out_range=(0, 1)),
            name='noisy_test',
        )
        viewer.add_image(
            rescale_intensity(denoised_test, in_range='image', out_range=(0, 1)),
            name='denoised_test',
        )


demo()
