import time
from os.path import join

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, add_noise
from aydin.io.folders import get_temp_folder
from aydin.it.base import ImageTranslatorBase
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.regression.lgbm import LGBMRegressor
from aydin.regression.linear import LinearRegressor
from aydin.regression.nn import NNRegressor
from aydin.regression.random_forest import RandomForestRegressor
from aydin.regression.support_vector import SupportVectorRegressor


@pytest.mark.heavy
def test_saveload_Linear():
    regressor = LinearRegressor()
    saveload(StandardFeatureGenerator(), regressor, min_psnr=19, min_ssim=0.73)


@pytest.mark.heavy
def test_saveload_RF():
    regressor = RandomForestRegressor()
    saveload(StandardFeatureGenerator(), regressor, min_ssim=0.75)


@pytest.mark.heavy
def test_saveload_SVR():
    regressor = SupportVectorRegressor()
    saveload(StandardFeatureGenerator(), regressor, min_psnr=22, min_ssim=0.71)


@pytest.mark.heavy
def test_saveload_NN():
    regressor = NNRegressor(max_epochs=12)
    saveload(StandardFeatureGenerator(), regressor, min_psnr=21, min_ssim=0.73)


def test_saveload_CB():
    regressor = CBRegressor(max_num_estimators=256, min_num_estimators=64)
    saveload(StandardFeatureGenerator(), regressor, min_ssim=0.78)


@pytest.mark.heavy
def test_saveload_LGBM():
    regressor = LGBMRegressor(max_num_estimators=256)
    saveload(StandardFeatureGenerator(), regressor, min_ssim=0.79)


def saveload(generator, regressor, min_psnr=22, min_ssim=0.75):
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.train(noisy, noisy)

    temp_file = join(get_temp_folder(), "test_it_saveload" + str(time.time()))
    print(temp_file)

    it.save(temp_file)
    del it

    loaded_it = ImageTranslatorBase.load(temp_file)

    denoised = loaded_it.translate(noisy)

    denoised = denoised.clip(0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
