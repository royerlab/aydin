import numpy as np
from napari.util import app_context
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise
from tifffile import imread

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.lgbm import LightGBMRegressor


<<<<<<< HEAD:src/pitl/examples/demo_pitl_2D_CARE_example.py
=======
def demo_pitl_2D():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = camera().astype(np.float32) #[:,50:450]
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    from napari import ViewerApp
    with app_context():
        viewer = ViewerApp()
        viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
        viewer.add_image(rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy')

        scales = [1, 3, 5, 11, 21, 23, 47, 95]
        widths = [3, 3, 3,  3,  3,  3,  3,  3]

        for param in range(7, len(scales), 1):

            generator = MultiscaleConvolutionalFeatures(kernel_widths=widths[0:param],
                                                        kernel_scales=scales[0:param],
                                                        kernel_shapes=['l1']*len(scales[0:param]),
                                                        exclude_center=True,
                                                        )

            regressor = LightGBMRegressor(  learning_rate=0.01,
                                            num_leaves=256,
                                            max_depth=8,
                                            n_estimators=1024,
                                            early_stopping_rounds=20)

            it = ImageTranslator(feature_generator=generator, regressor=regressor)

            denoised = it.train(noisy, noisy)
            # denoised_predict = pitl.predict(noisy)

            print("noisy", psnr(noisy, image), ssim(noisy, image))
            print("denoised", psnr(denoised, image), ssim(denoised, image))
            # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

            viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised%d' % param)
            # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


>>>>>>> master:src/pitl/test/demo_pitl_2D.py
def demo_pitl_2D_CARE_example():
    """
        Demo for supervised denoising using CARE example as a large 'montage'

        Get the data from here: https://drive.google.com/drive/folders/1-2QfKhWXSR-ulZrdhMPz_grjX4kT4d5_?usp=sharing
        put it in a folder 'data' at the root of the project (see below:)

        TODO: performance is not great in terms of image quality, does not performa as well as the original CPU version. Needs fixing.
              compare to results in 'sandbox_lightgbm_original.py'
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
        viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
        viewer.add_image(rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy')

        scales = [1, 3, 5, 11, 21, 23, 47, 95]
        widths = [3, 3, 3,  3,  3,  3,  3,  3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
                                                    exclude_center=False
                                                    )

        regressor = LightGBMRegressor(num_leaves=63,
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


<<<<<<< HEAD:src/pitl/examples/demo_pitl_2D_CARE_example.py
demo_pitl_2D_CARE_example()
=======
# Choose what to run here:

demo_pitl_2D()
#demo_pitl_2D_CARE_example()
>>>>>>> master:src/pitl/test/demo_pitl_2D.py
