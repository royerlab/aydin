"""Tests for regression models including fit, predict, and save/load."""

import time
from os.path import join

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.analysis.image_metrics import ssim
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.io.folders import get_temp_folder
from aydin.regression.base import RegressorBase
from aydin.regression.cb import CBRegressor
from aydin.regression.linear import LinearRegressor
from aydin.regression.perceptron import PerceptronRegressor
from aydin.regression.support_vector import SupportVectorRegressor

try:
    from aydin.regression.lgbm import LGBMRegressor
    from aydin.regression.random_forest import RandomForestRegressor

    _lgbm_available = True
except (ImportError, OSError):
    _lgbm_available = False


@pytest.fixture(scope="session")
def regressor_test_data():
    """Provide a noisy camera image with precomputed features for regressor tests."""
    image = camera()[:256, :256].astype(numpy.float32)
    image = normalise(image)
    noisy = add_noise(image)

    # feature generator requires images in 'standard' form: BCTZYX
    noisy = noisy[numpy.newaxis, numpy.newaxis, ...]

    generator = StandardFeatureGenerator(max_level=6)

    features = generator.compute(noisy, exclude_center_value=True)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    return image, noisy, x, y


_lgbm_param = (
    pytest.param(
        LGBMRegressor(max_num_estimators=600),
        0.60,
        marks=pytest.mark.skipif(not _lgbm_available, reason="LightGBM unavailable"),
    )
    if _lgbm_available
    else pytest.param(
        None,
        0.60,
        marks=pytest.mark.skip(reason="LightGBM unavailable (libomp missing?)"),
    )
)

_rf_param = (
    pytest.param(
        RandomForestRegressor(),
        0.55,
        marks=pytest.mark.skipif(
            not _lgbm_available, reason="LightGBM unavailable (RF depends on it)"
        ),
    )
    if _lgbm_available
    else pytest.param(
        None,
        0.55,
        marks=pytest.mark.skip(reason="LightGBM unavailable (RF depends on it)"),
    )
)


@pytest.mark.parametrize(
    "regressor, min_ssim",
    [
        (LinearRegressor(), 0.55),
        _rf_param,
        (SupportVectorRegressor(), 0.50),
        _lgbm_param,
        (CBRegressor(max_num_estimators=600), 0.60),
        (PerceptronRegressor(max_epochs=50, depth=6), 0.55),
    ],
)
def test_regressor(regressor, min_ssim, regressor_test_data):
    """Test that regressor denoising meets minimum SSIM and survives save/load."""
    image, noisy, x, y = regressor_test_data

    regressor.fit(x, y)

    yp = regressor.predict(x)

    denoised = yp.reshape(image.shape)

    denoised = numpy.clip(denoised, 0, 1)

    ssim_value = ssim(denoised, image)
    psnr_value = psnr(image, denoised)

    print("denoised", psnr_value, ssim_value)

    assert ssim_value > min_ssim

    # Test saveload
    temp_file = join(get_temp_folder(), "test_reg_saveload.json" + str(time.time()))
    regressor.save(temp_file)

    del regressor

    loaded_regressor = RegressorBase.load(temp_file)

    yp = loaded_regressor.predict(x)

    denoised = yp.reshape(image.shape)

    denoised = denoised.clip(0, 1)
    ssim_value = ssim(denoised, image)
    psnr_value = psnr(image, denoised)

    print("denoised", psnr_value, ssim_value)

    assert ssim_value > min_ssim


# --- Edge case tests ---


def test_regressor_repr():
    """Regressors should have reasonable string representations."""
    regressors = [CBRegressor(), LinearRegressor()]
    if _lgbm_available:
        regressors.append(LGBMRegressor())
    for reg in regressors:
        r = repr(reg)
        assert isinstance(r, str)
        assert len(r) > 0


def test_regressor_predict_before_fit():
    """Predicting before fitting should raise an error."""
    x = numpy.random.randn(10, 5).astype(numpy.float32)
    for reg in [LinearRegressor(), CBRegressor()]:
        with pytest.raises(Exception):
            reg.predict(x)


def test_regressor_constant_target():
    """Regressors should handle constant target without crashing."""
    x = numpy.random.randn(100, 5).astype(numpy.float32)
    y = numpy.ones(100, dtype=numpy.float32)
    reg = LinearRegressor()
    reg.fit(x, y)
    yp = reg.predict(x)
    # All predictions should be close to 1.0
    numpy.testing.assert_allclose(yp, 1.0, atol=0.1)


def test_regressor_single_feature():
    """Regressors should work with a single feature column."""
    x = numpy.random.randn(50, 1).astype(numpy.float32)
    y = (x[:, 0] * 2 + 1).astype(numpy.float32)
    reg = LinearRegressor()
    reg.fit(x, y)
    yp = reg.predict(x)
    # LinearRegressor may return (num_channels, N), flatten for comparison
    assert yp.flatten().shape == y.shape
