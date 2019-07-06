import numpy
from napari import Viewer
from napari.util import app_context
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from pitl.regression.gbm import GBMRegressor


def test_lgbm_regressor():
    display = False

    image = camera().astype(numpy.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    scales = [1, 3, 7, 15]
    widths = [3, 3, 3, 3]

    generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                kernel_scales=scales,
                                                kernel_shapes=['l1'] * len(scales),
                                                exclude_center=True,
                                                )

    regressor = GBMRegressor(learning_rate=0.01,
                             num_leaves=127,
                             max_bin=512,
                             n_estimators=512,
                             early_stopping_rounds=20)

    features = generator.compute(noisy)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    regressor.fit(x, y)

    yp = regressor.predict(x)

    denoised = yp.reshape(image.shape)

    ssim_value = ssim(denoised, image)
    psnr_value = psnr(denoised, image)

    print("denoised", psnr_value, ssim_value)

    if display:
        with app_context():
            viewer = Viewer()
            viewer.add_image(rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image')
            viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')

    assert ssim_value > 0.84
