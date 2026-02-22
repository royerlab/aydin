"""Tests for the Linear regressor.

Tests Linear regressor modes (linear, huber, lasso) and parameter handling.
"""

import numpy
import pytest
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.regression.linear import LinearRegressor


@pytest.fixture(scope='module')
def linear_test_data():
    """Provide feature matrix and target vector for linear regressor tests."""
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


@pytest.mark.parametrize('mode', ['linear', 'huber', 'lasso'])
def test_linear_all_modes(mode, linear_test_data):
    """Test all three linear regressor modes produce valid predictions."""
    x, y = linear_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LinearRegressor(mode=mode)
    regressor.fit(x[:split], y[:split])

    predictions = regressor.predict(x[split:])
    # predict returns (n_channels, n_samples) — single channel here
    assert predictions.shape[-1] == n - split
    assert not numpy.isnan(predictions).any()


def test_linear_default_mode():
    """Test default mode is huber."""
    regressor = LinearRegressor()
    assert regressor.mode == 'huber'


def test_linear_lasso_alpha_parameter(linear_test_data):
    """Test that Lasso alpha parameter is used."""
    x, y = linear_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LinearRegressor(mode='lasso', alpha=0.01)
    regressor.fit(x[:split], y[:split])

    predictions = regressor.predict(x[split:])
    assert predictions.shape[-1] == n - split


def test_linear_model_save_load_roundtrip(linear_test_data, tmp_path):
    """Test save/load roundtrip produces consistent predictions."""
    x, y = linear_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = LinearRegressor(mode='linear')
    regressor.fit(x[:split], y[:split])

    predictions_before = regressor.predict(x[split:])

    save_path = str(tmp_path / 'linear_model')
    regressor.save(save_path)

    loaded = LinearRegressor.load(save_path)
    predictions_after = loaded.predict(x[split:])

    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)
