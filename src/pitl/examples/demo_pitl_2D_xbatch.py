import time

import numpy as np
from napari import Viewer
from napari.util import app_context
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_2D(image, min_level=7, max_level=100):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with app_context():
        viewer = Viewer()
        viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
        viewer.add_image(rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy')

        scales = [1, 3, 7, 15, 31, 63, 127, 255]
        widths = [3, 3, 3,  3,  3,  3,   3,   3]

        for param in range(min_level, min(max_level,len(scales)), 1):
            generator = MultiscaleConvolutionalFeatures(kernel_widths=widths[0:param],
                                                        kernel_scales=scales[0:param],
                                                        kernel_shapes=['l1'] * len(scales[0:param]),
                                                        exclude_center=True,
                                                        )

            regressor = GBMRegressor(learning_rate=0.01,
                                     num_leaves=256,
                                     max_depth=8,
                                     n_estimators=2048,
                                     early_stopping_rounds=20)

            it = ImageTranslator(feature_generator=generator, regressor=regressor)

            start = time.time()
            denoised = it.train(noisy, noisy, batch_dims=(False, True))
            stop = time.time()
            print(f"Training: elapsed time:  {stop-start} ")
            # denoised_predict = pitl.predict(noisy)

            print("noisy", psnr(noisy, image), ssim(noisy, image))
            print("denoised", psnr(denoised, image), ssim(denoised, image))
            # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

            viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised%d' % param)
            # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo_pitl_2D(camera().astype(np.float32), min_level=7)

# for example in examples_single.get_list():
#     example_file_path = examples_single.get_path(*example)
#     array, metadata = io.imread(example_file_path)
#     demo_pitl_2D(array.astype(np.float32), min_level=5, max_level=6)
