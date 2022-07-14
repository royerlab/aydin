import time

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, add_noise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.regression.lgbm import LGBMRegressor
from aydin.regression.linear import LinearRegressor
from aydin.regression.perceptron import PerceptronRegressor
from aydin.regression.random_forest import RandomForestRegressor
from aydin.regression.support_vector import SupportVectorRegressor


@pytest.mark.heavy
def test_it_fgr_linear():
    regressor = LinearRegressor()
    do_it_fgr(regressor, min_psnr=18, min_ssim=0.65)


def test_it_fgr_rf():
    regressor = RandomForestRegressor(max_num_estimators=128)
    do_it_fgr(regressor, min_ssim=0.75)


@pytest.mark.heavy
def test_it_fgr_svr():
    regressor = SupportVectorRegressor()
    do_it_fgr(regressor, min_psnr=22, min_ssim=0.71)


def test_it_fgr_lgbm():
    regressor = LGBMRegressor(max_num_estimators=256)
    do_it_fgr(regressor, min_ssim=0.77)


@pytest.mark.heavy
def test_it_fgr_nn():
    regressor = PerceptronRegressor(max_epochs=64)
    do_it_fgr(regressor, min_ssim=0.70)


@pytest.mark.heavy  # we skip this as we run same for saveload
def test_it_fgr_cb():
    regressor = CBRegressor(max_num_estimators=256, min_num_estimators=64)
    do_it_fgr(regressor, min_ssim=0.779)


def test_it_fgr_cb_supervised():
    regressor = CBRegressor(max_num_estimators=256, min_num_estimators=64)
    do_it_fgr(regressor, min_ssim=0.77, supervised=True)


def do_it_fgr(regressor, min_psnr=22, min_ssim=0.75, supervised=False):
    """
    Test for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    if supervised:
        generator = StandardFeatureGenerator(include_spatial_features=False)
    else:
        generator = StandardFeatureGenerator(max_level=5, include_spatial_features=True)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    if supervised:
        it.train(noisy, image)
    else:
        it.train(noisy, noisy)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop - start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop - start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the regressor have been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
