from os.path import join

import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from pitl.io.examples import downloaded_zipped_example, zipped_datasets
from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_2D_CARE_example():

    downloaded_zipped_example('tribolium')

    image = imread(join(zipped_datasets.get_path(*zipped_datasets.care_tribolium), 'tribolium_train_GT_montage.tif')).astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    noisy = imread(join(zipped_datasets.get_path(*zipped_datasets.care_tribolium), 'tribolium_train_low_montage.tif')).astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    image_test = imread(join(zipped_datasets.get_path(*zipped_datasets.care_tribolium), 'tribolium_test_GT_montage.tif')).astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1))

    noisy_test = imread(join(zipped_datasets.get_path(*zipped_datasets.care_tribolium), 'tribolium_test_low_montage.tif')).astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1))

    from napari import ViewerApp
    with app_context():
        viewer = ViewerApp()
        viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
        viewer.add_image(rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy')

        scales = [1, 3, 5, 11, 21, 23, 47, 95]
        widths = [3, 3, 3,  3,  3,  3,  3,  3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(num_leaves=63,
                                 n_estimators=512)

        it = ImageTranslator(feature_generator=generator, regressor=regressor)

        denoised = it.train(noisy, image)
        denoised_test = it.translate(noisy_test)

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')

        print("denoised_test", psnr(denoised_test, image_test), ssim(denoised_test, image_test))
        viewer.add_image(rescale_intensity(image_test, in_range='image', out_range=(0, 1)), name='image_test')
        viewer.add_image(rescale_intensity(noisy_test, in_range='image', out_range=(0, 1)), name='noisy_test')
        viewer.add_image(rescale_intensity(denoised_test, in_range='image', out_range=(0, 1)), name='denoised_test')


demo_pitl_2D_CARE_example()

