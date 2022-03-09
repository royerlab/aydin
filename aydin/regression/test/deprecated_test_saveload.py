import time
from os.path import join

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.io.folders import get_temp_folder
from aydin.regression.base import RegressorBase
from aydin.regression.cb import CBRegressor
from aydin.regression.lgbm import LGBMRegressor
from aydin.regression.linear import LinearRegressor
from aydin.regression.nn import NNRegressor
from aydin.regression.random_forest import RandomForestRegressor
from aydin.regression.support_vector import SupportVectorRegressor


def test_linear_saveload():
    saveload(LinearRegressor(), min_ssim=0.70)


def test_rf_saveload():
    saveload(RandomForestRegressor(), min_ssim=0.70)


def test_svr_saveload():
    saveload(SupportVectorRegressor(), min_ssim=0.70)


def test_gbm_saveload():
    saveload(LGBMRegressor(max_num_estimators=128), min_ssim=0.73)


def test_cb_saveload():
    saveload(CBRegressor(max_num_estimators=128, min_num_estimators=64), min_ssim=0.72)


def test_nn_saveload():
    saveload(NNRegressor(max_epochs=10), min_ssim=0.71)


def saveload(regressor, min_ssim=0.80):

    image = camera().astype(numpy.float32)
    image = normalise(image)
    noisy = add_noise(image)

    # feature generator requires images in 'standard' form: BCTZYX
    noisy = noisy[numpy.newaxis, numpy.newaxis, ...]

    generator = StandardFeatureGenerator()

    features = generator.compute(noisy, exclude_center_value=True)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    regressor.fit(x, y)

    temp_file = join(get_temp_folder(), "test_reg_saveload.json" + str(time.time()))
    regressor.save(temp_file)

    del regressor

    loaded_regressor = RegressorBase.load(temp_file)

    yp = loaded_regressor.predict(x)

    denoised = yp.reshape(image.shape)

    # print(numpy.min(denoised), numpy.min(image))
    # print(numpy.max(denoised), numpy.max(image))
    denoised = denoised.clip(0, 1)
    ssim_value = ssim(denoised, image)
    psnr_value = psnr(denoised, image)

    print("denoised", psnr_value, ssim_value)

    assert ssim_value > min_ssim
