"""Tests for the RandomForestRegressor.

Tests RF-specific behavior (boosting_type='rf', feature_fraction) plus
standard fit/predict/save-load cycle.
"""

import numpy
import pytest
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.regression.random_forest import RandomForestRegressor


@pytest.fixture(scope='module')
def rf_test_data():
    """Provide feature matrix and target vector for RF tests."""
    image = normalise(camera()[:256, :256].astype(numpy.float32))
    noisy = add_noise(image)

    noisy_std = noisy[numpy.newaxis, numpy.newaxis, ...]

    generator = StandardFeatureGenerator(max_level=6)
    features = generator.compute(noisy_std, exclude_center_value=True)
    x = features.reshape(-1, features.shape[-1])
    y = noisy_std.reshape(-1)

    rng = numpy.random.RandomState(42)
    indices = rng.choice(len(x), size=min(20000, len(x)), replace=False)
    return x[indices], y[indices]


def test_rf_default_construction():
    """Test default constructor values for RandomForestRegressor."""
    regressor = RandomForestRegressor()
    assert regressor.num_leaves == 1024
    assert regressor.max_num_estimators == 2048
    assert regressor.learning_rate == 0.0001
    assert regressor.metric == 'l1'


def test_rf_repr():
    """Test string representation."""
    regressor = RandomForestRegressor()
    r = repr(regressor)
    assert 'RandomForestRegressor' in r
    assert 'max_num_estimators' in r


def test_rf_get_params_boosting_type():
    """Test that _get_params sets boosting_type to 'rf' and feature_fraction."""
    regressor = RandomForestRegressor()
    params = regressor._get_params(num_samples=1000)
    assert params['boosting_type'] == 'rf'
    assert params['feature_fraction'] == 0.8


def test_rf_fit_predict(rf_test_data):
    """Test fit/predict cycle produces valid predictions."""
    x, y = rf_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = RandomForestRegressor(max_num_estimators=50)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    assert predictions.shape[-1] == n - split
    assert not numpy.isnan(predictions).any()


def test_rf_save_load_roundtrip(rf_test_data, tmp_path):
    """Test save/load roundtrip produces consistent predictions."""
    x, y = rf_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = RandomForestRegressor(max_num_estimators=50)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions_before = regressor.predict(x[split:])

    save_path = str(tmp_path / 'rf_model')
    regressor.save(save_path)

    loaded = RandomForestRegressor.load(save_path)
    predictions_after = loaded.predict(x[split:])

    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)
