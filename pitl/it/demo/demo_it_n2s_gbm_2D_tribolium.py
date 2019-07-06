import time
from os.path import join

import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from pitl.io.datasets import examples_zipped
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor
from pitl.features.mcfocl import MultiscaleConvolutionalFeatures


def demo():
    """
        Demo for supervised denoising using CARE's tribolium example as a montage.

    """

    image = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_GT_montage.tif')).astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    noisy = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_montage.tif')).astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    image_test = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_GT_montage.tif')).astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1))

    noisy_test = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_montage.tif')).astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1))

    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
        viewer.add_image(rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy')

        scales = [1, 3, 7, 15, 31]
        widths = [3, 3, 3, 3, 3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(metric='poisson',
                                 learning_rate=0.01,
                                 num_leaves=127,
                                 max_bin=512,
                                 n_estimators=2048,
                                 early_stopping_rounds=20)

        it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

        start = time.time()
        denoised = it.train(noisy, image)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised_test = it.translate(noisy_test)
        stop = time.time()
        print(f"inference: elapsed time:  {stop-start} ")

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        print("denoised_test", psnr(denoised_test, image_test), ssim(denoised_test, image_test))

        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')
        viewer.add_image(rescale_intensity(image_test, in_range='image', out_range=(0, 1)), name='image_test')
        viewer.add_image(rescale_intensity(noisy_test, in_range='image', out_range=(0, 1)), name='noisy_test')
        viewer.add_image(rescale_intensity(denoised_test, in_range='image', out_range=(0, 1)), name='denoised_test')


demo()
