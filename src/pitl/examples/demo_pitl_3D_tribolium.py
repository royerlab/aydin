from os.path import join

import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from src.pitl.io.datasets import downloaded_zipped_example, examples_zipped
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_3D():
    """
        Demo for supervised denoising using CARE's tribolium example -- full 3D.

    """

    downloaded_zipped_example('tribolium')

    image = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_GT_stack.tif')).astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1)).astype(np.float32)[:, 200:600, 200:400]

    noisy = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_stack.tif')).astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1)).astype(np.float32)[:, 200:600, 200:400]

    image_test = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_GT_stack.tif')).astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1)).astype(np.float32)

    noisy_test = imread(join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_stack.tif')).astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1)).astype(np.float32)

    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(image_test, name='image_test')
        viewer.add_image(noisy_test, name='noisy_test')

        level = 6
        scales = [1, 3, 7, 15, 31, 63, 127, 255]
        widths = [3, 3, 3,  3,  3,  3,  3,  3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths[:level],
                                                    kernel_scales=scales[:level],
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(num_leaves=64,
                                 max_depth=7,
                                 n_estimators=1024,
                                 learning_rate=0.01,
                                 eval_metric='l1',
                                 early_stopping_rounds=None)

        it = ImageTranslator(generator, regressor)

        denoised = it.train(noisy, image)
        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')

        image_test_denoised = it.translate(noisy_test)
        viewer.add_image(rescale_intensity(image_test_denoised, in_range='image', out_range=(0, 1)), name='test_denoised')

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        print("denoised test", psnr(image_test_denoised, image_test), ssim(image_test_denoised, image_test))


demo_pitl_3D()
