import time

import numpy as np
from napari.util import app_context
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_2D():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = camera().astype(np.float32)  # [:,50:450]
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
        widths = [7, 3, 3, 3, 3, 3, 3, 3]

        for param in range(7, len(scales), 1):
            generator = MultiscaleConvolutionalFeatures(kernel_widths=widths[0:param],
                                                        kernel_scales=scales[0:param],
                                                        kernel_shapes=['l1'] * len(scales[0:param]),
                                                        exclude_center=True,
                                                        )

            regressor = GBMRegressor(learning_rate=0.01,
                                     num_leaves=256,
                                     max_depth=8,
                                     n_estimators=2024,
                                     early_stopping_rounds=20)

            it = ImageTranslator(feature_generator=generator, regressor=regressor)

            start = time.time()
            denoised = it.train(noisy, noisy)
            stop = time.time()
            print(f"Training: elapsed time:  {stop-start} ")
            # denoised_predict = pitl.predict(noisy)

            print("noisy", psnr(noisy, image), ssim(noisy, image))
            print("denoised", psnr(denoised, image), ssim(denoised, image))
            # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

            viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised%d' % param)
            # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo_pitl_2D()
