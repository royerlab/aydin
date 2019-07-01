import time

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor


def test_it_classic():
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(camera().astype(numpy.float32), in_range='image', out_range=(0, 1))

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = MultiscaleConvolutionalFeatures(exclude_center=True)

    regressor = GBMRegressor(learning_rate=0.01,
                             num_leaves=127,
                             max_bin=512,
                             n_estimators=2048,
                             early_stopping_rounds=20)

    it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

    start = time.time()
    denoised = it.train(noisy, noisy)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised_predict = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    psnr_denoised_inf = psnr(denoised_predict, image)
    ssim_denoised_inf = ssim(denoised_predict, image)
    print("denoised_predict", psnr_denoised_inf, ssim_denoised_inf)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert abs(psnr_denoised - psnr_denoised_inf) < 0.1 and abs(ssim_denoised - ssim_denoised_inf) < 0.01

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 24 and ssim_denoised > 0.84
