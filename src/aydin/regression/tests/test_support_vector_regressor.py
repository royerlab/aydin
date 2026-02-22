"""Tests for the SupportVectorRegressor.

SVR is slow, so tests use small data and the linear variant.
The RBF test is marked heavy.
"""

import numpy
import pytest

from aydin.regression.support_vector import SupportVectorRegressor, _SVRModel


def _make_small_data(n_samples=200, n_features=5, seed=42):
    """Generate a small regression dataset."""
    rng = numpy.random.RandomState(seed)
    x = rng.randn(n_samples, n_features).astype(numpy.float32)
    y = (x[:, 0] * 2 + x[:, 1] * 0.5 + rng.randn(n_samples) * 0.1).astype(numpy.float32)
    return x, y


def test_svr_default_construction():
    """Test default constructor creates linear SVR."""
    reg = SupportVectorRegressor()
    assert reg.linear is True


def test_svr_repr():
    """Test string representation."""
    reg = SupportVectorRegressor(linear=True)
    r = repr(reg)
    assert 'SupportVectorRegressor' in r
    assert 'linear=True' in r


def test_svr_linear_fit_predict():
    """Test LinearSVR fit/predict cycle."""
    x, y = _make_small_data()
    split = 160

    reg = SupportVectorRegressor(linear=True)
    reg.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = reg.predict(x[split:])
    assert predictions.shape[-1] == len(x) - split
    assert not numpy.isnan(predictions).any()


def test_svr_save_load_roundtrip(tmp_path):
    """Test save/load roundtrip produces consistent predictions."""
    x, y = _make_small_data()
    split = 160

    reg = SupportVectorRegressor(linear=True)
    reg.fit(x[:split], y[:split], x[split:], y[split:])

    predictions_before = reg.predict(x[split:])

    save_path = str(tmp_path / 'svr_model')
    reg.save(save_path)

    from aydin.regression.base import RegressorBase

    loaded = RegressorBase.load(save_path)
    predictions_after = loaded.predict(x[split:])

    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)


def test_svr_model_loss_history():
    """SVR model wrapper has empty loss_history dict."""
    from sklearn.svm import LinearSVR

    model = LinearSVR()
    x, y = _make_small_data(n_samples=50)
    model.fit(x, y)

    wrapper = _SVRModel(model)
    assert 'training' in wrapper.loss_history
    assert 'validation' in wrapper.loss_history
    assert len(wrapper.loss_history['training']) == 0


def test_svr_model_save_load_internals_noop(tmp_path):
    """SVR model internals save/load are no-ops and don't raise."""
    from sklearn.svm import LinearSVR

    model = LinearSVR()
    x, y = _make_small_data(n_samples=50)
    model.fit(x, y)

    wrapper = _SVRModel(model)
    wrapper._save_internals(str(tmp_path))
    wrapper._load_internals(str(tmp_path))


@pytest.mark.heavy
def test_svr_rbf_fit_predict():
    """Test RBF SVR fit/predict cycle (slower)."""
    x, y = _make_small_data(n_samples=100)
    split = 80

    reg = SupportVectorRegressor(linear=False)
    reg.fit(x[:split], y[:split], x[split:], y[split:])

    predictions = reg.predict(x[split:])
    assert predictions.shape[-1] == len(x) - split
    assert not numpy.isnan(predictions).any()
