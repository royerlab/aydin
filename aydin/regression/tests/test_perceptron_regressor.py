"""Tests for the Perceptron (MLP) regressor.

Tests PyTorch-based perceptron functionality including loss functions,
integer dtype handling, early stopping, and model persistence.
"""

import numpy
import pytest
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, normalise
from aydin.regression.perceptron import PerceptronRegressor


@pytest.fixture(scope='module')
def perceptron_test_data():
    """Provide feature matrix and target vector for perceptron tests."""
    image = normalise(camera()[:256, :256].astype(numpy.float32))
    noisy = add_noise(image)

    # Feature generator requires images in 'standard' form: BCTZYX
    noisy_std = noisy[numpy.newaxis, numpy.newaxis, ...]

    generator = StandardFeatureGenerator(max_level=6)
    features = generator.compute(noisy_std, exclude_center_value=True)
    x = features.reshape(-1, features.shape[-1])
    y = noisy_std.reshape(-1)

    # Subsample heavily for speed (perceptron is slow)
    rng = numpy.random.RandomState(42)
    indices = rng.choice(len(x), size=min(5000, len(x)), replace=False)
    return x[indices], y[indices]


def test_perceptron_default_construction():
    """Test default constructor values."""
    regressor = PerceptronRegressor()
    assert regressor.max_epochs == 1024
    assert regressor.learning_rate == 0.001
    assert regressor.depth == 6
    # 'l1' gets mapped to 'mae' internally
    assert regressor.loss == 'mae'


@pytest.mark.parametrize('loss', ['l1', 'l2'])
def test_perceptron_both_losses(loss, perceptron_test_data):
    """Test perceptron with l1 and l2 losses."""
    x, y = perceptron_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = PerceptronRegressor(loss=loss, max_epochs=10, depth=3)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    # Perceptron predict returns (1, n_samples, 1) for single-channel
    n_predicted = predictions.flatten().shape[0]
    assert n_predicted == n - split
    assert not numpy.isnan(predictions).any()


def test_perceptron_early_stopping(perceptron_test_data):
    """Test that perceptron stops early with small patience."""
    x, y = perceptron_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = PerceptronRegressor(max_epochs=100, patience=2, depth=3)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = regressor.predict(x[split:])
    n_predicted = predictions.flatten().shape[0]
    assert n_predicted == n - split


def test_perceptron_integer_dtype_rescaling():
    """Test that perceptron handles uint8 input correctly."""
    rng = numpy.random.RandomState(42)
    x = rng.random((2000, 10)).astype(numpy.float32)
    y_uint8 = rng.randint(0, 256, size=2000).astype(numpy.uint8)

    regressor = PerceptronRegressor(max_epochs=5, depth=3)
    regressor.fit(x[:1600], y_uint8[:1600], x[1600:], y_uint8[1600:])

    predictions = regressor.predict(x[1600:])
    n_predicted = predictions.flatten().shape[0]
    assert n_predicted == 400


def test_perceptron_model_save_load(perceptron_test_data, tmp_path):
    """Test PyTorch model persistence."""
    x, y = perceptron_test_data
    n = len(x)
    split = int(0.8 * n)

    regressor = PerceptronRegressor(max_epochs=10, depth=3)
    regressor.fit(x[:split], y[:split], x[split:], y[split:])

    predictions_before = regressor.predict(x[split:]).flatten()

    save_path = str(tmp_path / 'perceptron_model')
    regressor.save(save_path)

    loaded = PerceptronRegressor.load(save_path)
    predictions_after = loaded.predict(x[split:]).flatten()

    numpy.testing.assert_array_almost_equal(
        predictions_before, predictions_after, decimal=4
    )
