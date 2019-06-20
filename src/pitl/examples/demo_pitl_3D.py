import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_3D():
    """
        Demo for supervised denoising using CARE example.

        Get the data from here: https://drive.google.com/drive/folders/1-2QfKhWXSR-ulZrdhMPz_grjX4kT4d5_?usp=sharing
        put it in a folder 'data' at the root of the project (see below:)
    """

    image = imread('../../../data/tribolium/train/GT/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    noisy = imread('../../../data/tribolium/train/low/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    image_test = imread('../../../data/tribolium/test/GT/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1))

    noisy_test = imread('../../../data/tribolium/test/low/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1))

    from napari import ViewerApp
    with app_context():
        viewer = ViewerApp()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(image_test, name='image_test')
        viewer.add_image(noisy_test, name='noisy_test')

        scales = [1, 3, 5, 11, 21, 23, 47, 95]
        widths = [3, 3, 3,  3,  3,  3,  3,  3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
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
