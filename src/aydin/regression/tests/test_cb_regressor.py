"""Tests for the CatBoost regressor.

Tests CatBoost-specific functionality including GPU/CPU configuration,
loss functions, early stopping, and model persistence.
"""

import numpy
import pytest
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.regression.cb import CBRegressor


@pytest.fixture(scope='module')
def cb_test_data():
    """Provide feature matrix and target vector for CatBoost tests."""
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


def test_cb_recommended_max_datapoints():
    """Test that recommended_max_num_datapoints returns a positive value."""
    regressor = CBRegressor()
    max_points = regressor.recommended_max_num_datapoints()
    assert max_points > 0


def test_cb_default_construction():
    """Test default constructor values."""
    regressor = CBRegressor()
    assert regressor.metric == 'l1'
    assert regressor.early_stopping_rounds == 32
    assert regressor.compute_load == 0.95


@pytest.mark.parametrize('loss', ['l1', 'l2'])
def test_cb_different_loss_functions(loss, cb_test_data):
    """Test CatBoost with different loss functions."""
    x, y = cb_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = CBRegressor(
        loss=loss, max_num_estimators=50, min_num_estimators=10, gpu=False
    )
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    # predict returns (n_channels, n_samples) — single channel here
    assert predictions.shape[-1] == n - split
    assert not numpy.isnan(predictions).any()


def test_cb_early_stopping(cb_test_data):
    """Test that CatBoost uses early stopping during training."""
    x, y = cb_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = CBRegressor(
        max_num_estimators=500, min_num_estimators=10, patience=5, gpu=False
    )
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    assert predictions.shape[-1] == n - split


def test_cb_loss_history_recording(cb_test_data):
    """Test that loss_history is populated after fitting."""
    x, y = cb_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = CBRegressor(max_num_estimators=30, min_num_estimators=10, gpu=False)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    assert len(regressor.loss_history) > 0


def test_cb_model_save_load_roundtrip(cb_test_data, tmp_path):
    """Test save/load roundtrip produces consistent predictions."""
    x, y = cb_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = CBRegressor(max_num_estimators=30, min_num_estimators=10, gpu=False)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions_before = regressor.predict(x[split:])

    save_path = str(tmp_path / 'cb_model')
    regressor.save(save_path)

    loaded = CBRegressor.load(save_path)
    predictions_after = loaded.predict(x[split:])

    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)
