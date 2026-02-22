"""Tests for the ImageTranslatorFGR (Feature Generation and Regression)."""

import time

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.analysis.image_metrics import ssim
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.it.fgr import ImageTranslatorFGR
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


@pytest.mark.heavy
@pytest.mark.parametrize(
    "regressor, min_psnr, min_ssim",
    [
        (LinearRegressor(), 18, 0.65),
        (SupportVectorRegressor(), 22, 0.71),
        (PerceptronRegressor(max_epochs=64), 22, 0.70),
        (CBRegressor(max_num_estimators=256, min_num_estimators=64), 22, 0.779),
    ],
)
def test_it_fgr_linear(regressor, min_psnr, min_ssim):
    """Test FGR with various heavy regressors (linear, SVM, perceptron, CB)."""
    do_it_fgr(regressor, min_psnr=min_psnr, min_ssim=min_ssim)


@pytest.mark.skipif(
    not _lgbm_available, reason="LightGBM unavailable (RF depends on it)"
)
def test_it_fgr_rf():
    """Test FGR with random forest regressor."""
    regressor = RandomForestRegressor(max_num_estimators=128)
    do_it_fgr(regressor, min_ssim=0.60)


@pytest.mark.skipif(not _lgbm_available, reason="LightGBM unavailable (libomp?)")
def test_it_fgr_lgbm():
    """Test FGR with LightGBM regressor."""
    regressor = LGBMRegressor(max_num_estimators=256)
    do_it_fgr(regressor, min_ssim=0.60)


def test_it_fgr_cb_supervised():
    """Test FGR with CatBoost regressor in supervised mode."""
    regressor = CBRegressor(max_num_estimators=256, min_num_estimators=64)
    do_it_fgr(regressor, min_ssim=0.65, supervised=True)


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

    # if the line below fails, then the parameters of the
    # image the regressor have been broken. do not change the
    # number below, but instead, fix the problem --
    # most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim


def test_fgr_stores_shape_normaliser_after_train():
    """Test that shape_normaliser is available as instance attr after training."""
    gen = StandardFeatureGenerator(include_spatial_features=False, max_level=2)
    it = ImageTranslatorFGR(feature_generator=gen)
    image = numpy.random.rand(64, 64).astype(numpy.float32)
    it.train(image, image)
    assert hasattr(it, 'shape_normaliser')
    assert it.shape_normaliser is not None
