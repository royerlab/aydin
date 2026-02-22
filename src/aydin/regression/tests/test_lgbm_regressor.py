"""Tests for the LightGBM regressor.

Tests LightGBM-specific functionality including loss functions, early stopping,
training loss tracking, and model persistence.
"""

import numpy
import pytest
from skimage.data import camera

try:
    from aydin.regression.lgbm import LGBMRegressor
except (ImportError, OSError) as _err:
    pytest.skip(f"LightGBM not available: {_err}", allow_module_level=True)

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise


@pytest.fixture(scope='module')
def lgbm_test_data():
    """Provide feature matrix and target vector for LightGBM tests."""
    image = normalise(camera()[:256, :256].astype(numpy.float32))
    noisy = add_noise(image)

    # Feature generator requires images in 'standard' form: BCTZYX
    noisy_std = noisy[numpy.newaxis, numpy.newaxis, ...]

    generator = StandardFeatureGenerator(max_level=6)
    features = generator.compute(noisy_std, exclude_center_value=True)
    x = features.reshape(-1, features.shape[-1])
    y = noisy_std.reshape(-1)

    # Subsample for speed
    rng = numpy.random.RandomState(42)
    indices = rng.choice(len(x), size=min(20000, len(x)), replace=False)
    return x[indices], y[indices]


def test_lgbm_default_construction():
    """Test default constructor values."""
    regressor = LGBMRegressor()
    assert regressor.metric == 'l1'
    assert regressor.early_stopping_rounds == 5
    assert regressor.max_bin == 512


@pytest.mark.parametrize('loss', ['l1', 'l2', 'huber'])
def test_lgbm_different_loss_functions(loss, lgbm_test_data):
    """Test LightGBM with different loss functions."""
    x, y = lgbm_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LGBMRegressor(loss=loss, max_num_estimators=50)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    # predict returns (n_channels, n_samples) — single channel here
    assert predictions.shape[-1] == n - split
    assert not numpy.isnan(predictions).any()


def test_lgbm_early_stopping(lgbm_test_data):
    """Test that LightGBM uses early stopping during training."""
    x, y = lgbm_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LGBMRegressor(max_num_estimators=500, patience=3)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    assert predictions.shape[-1] == n - split


def test_lgbm_loss_history_with_training_loss(lgbm_test_data):
    """Test loss history with compute_training_loss enabled."""
    x, y = lgbm_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LGBMRegressor(max_num_estimators=30, compute_training_loss=True)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    assert len(regressor.loss_history) > 0


def test_lgbm_uint8_max_bin_adjustment():
    """Test that max_bin is capped for uint8 inputs."""
    # Create uint8 features
    rng = numpy.random.RandomState(42)
    x = rng.randint(0, 256, size=(1000, 10)).astype(numpy.uint8)
    y = rng.random(1000).astype(numpy.float32)

    regressor = LGBMRegressor(max_bin=512, max_num_estimators=10)
    # The fit should handle uint8 data without error
    regressor.fit(x[:800], y[:800], x[800:], y[800:])
    predictions = regressor.predict(x[800:])
    assert predictions.shape[-1] == 200


def test_lgbm_model_save_load_roundtrip(lgbm_test_data, tmp_path):
    """Test save/load roundtrip produces consistent predictions."""
    x, y = lgbm_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LGBMRegressor(max_num_estimators=30)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions_before = regressor.predict(x[split:])

    save_path = str(tmp_path / 'lgbm_model')
    regressor.save(save_path)

    loaded = LGBMRegressor.load(save_path)
    predictions_after = loaded.predict(x[split:])

    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)
