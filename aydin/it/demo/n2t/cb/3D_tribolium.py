# flake8: noqa
import time
from os.path import join

import napari
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tifffile import imread

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import downloaded_zipped_example, examples_zipped, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def demo():
    """
    Demo for supervised denoising using CARE's tribolium example -- full 3D.

    Note: works quite well, but requires a serious NVIDIA graphics card...

    """

    downloaded_zipped_example('tribolium')

    image = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_gt_stack.tif')
    ).astype(np.float32)
    image = normalise(image)[:, 200:600, 200:400]

    noisy = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_stack.tif')
    ).astype(np.float32)
    noisy = normalise(noisy)[:, 200:600, 200:400]

    image_test = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_gt_stack.tif')
    ).astype(np.float32)
    image_test = normalise(image_test)

    noisy_test = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_stack.tif')
    ).astype(np.float32)
    noisy_test = normalise(noisy_test)

    generator = StandardFeatureGenerator(
        kernel_widths=[11, 9, 7, 5, 1, 1],  # , 1, 1, 1, 1],
        kernel_scales=[1, 3, 5, 9, 95, 191],  # , 11, 31, 71, 151],
        kernel_shapes=[
            'l1',
            'l1',
            'l1',
            'l1',
            'l1',
            'l1',
        ],  # , 'l1', 'l1', 'l1', 'l1'],
        include_scale_one=True,
    )
    regressor = CBRegressor()

    it = ImageTranslatorFGR(generator, regressor, max_voxels_for_training=1e7)

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
        "denoised test",
        psnr(denoised_test, image_test),
        ssim(denoised_test, image_test),
    )

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        viewer.add_image(image_test, name='image_test')
        viewer.add_image(noisy_test, name='noisy_test')
        viewer.add_image(denoised_test, name='test_denoised')


demo()
