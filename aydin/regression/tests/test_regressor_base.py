"""Tests for the RegressorBase abstract class.

Tests common infrastructure (multi-channel fit, stop_fit, getstate,
predict with models_to_use) using a minimal concrete subclass.
"""

import numpy
import pytest

from aydin.regression.base import RegressorBase


class _DummyModel:
    """Minimal model returned by _DummyRegressor._fit."""

    def __init__(self, offset=0.0):
        self.offset = offset
        self.loss_history = {'training': [0.1], 'validation': [0.2]}

    def predict(self, x):
        return x[:, 0] + self.offset

    def _save_internals(self, path):
        pass

    def _load_internals(self, path):
        pass


class _DummyRegressor(RegressorBase):
    """Concrete subclass of RegressorBase for testing."""

    def __init__(self):
        super().__init__()
        self._fit_count = 0

    def _fit(self, x_train, y_train, x_valid, y_valid, regressor_callback=None):
        self._fit_count += 1
        return _DummyModel(offset=self._fit_count)

    def _save_internals(self, path):
        pass

    def _load_internals(self, path):
        pass


def test_recommended_max_num_datapoints():
    """Default recommended_max_num_datapoints returns a large positive int."""
    reg = _DummyRegressor()
    max_pts = reg.recommended_max_num_datapoints()
    assert isinstance(max_pts, int)
    assert max_pts > 0


def test_stop_fit_sets_flag():
    """stop_fit() sets the internal _stop_fit flag to True."""
    reg = _DummyRegressor()
    assert reg._stop_fit is False
    reg.stop_fit()
    assert reg._stop_fit is True


def test_getstate_excludes_models():
    """__getstate__ should remove the 'models' key for serialization."""
    reg = _DummyRegressor()
    x = numpy.random.randn(20, 3).astype(numpy.float32)
    y = numpy.random.randn(20).astype(numpy.float32)
    reg.fit(x, y)

    state = reg.__getstate__()
    assert 'models' not in state
    assert 'num_channels' in state


def test_single_channel_fit():
    """fit() with 1D y trains one model."""
    reg = _DummyRegressor()
    x = numpy.random.randn(50, 3).astype(numpy.float32)
    y = numpy.random.randn(50).astype(numpy.float32)

    reg.fit(x, y)

    assert reg.num_channels == 1
    assert len(reg.models) == 1


def test_multi_channel_fit():
    """fit() with 2D y trains one model per channel."""
    reg = _DummyRegressor()
    x = numpy.random.randn(50, 3).astype(numpy.float32)
    y = numpy.random.randn(3, 50).astype(numpy.float32)  # 3 channels

    reg.fit(x, y)

    assert reg.num_channels == 3
    assert len(reg.models) == 3


def test_multi_channel_predict():
    """predict() returns stacked results for multi-channel model."""
    reg = _DummyRegressor()
    x = numpy.random.randn(50, 3).astype(numpy.float32)
    y = numpy.random.randn(2, 50).astype(numpy.float32)

    reg.fit(x, y)
    predictions = reg.predict(x)

    assert predictions.shape == (2, 50)


def test_predict_with_models_to_use():
    """predict() with models_to_use uses only the specified subset."""
    reg = _DummyRegressor()
    x = numpy.random.randn(30, 3).astype(numpy.float32)
    y = numpy.random.randn(3, 30).astype(numpy.float32)

    reg.fit(x, y)

    # Use only the first two models
    predictions = reg.predict(x, models_to_use=reg.models[:2])
    assert predictions.shape == (2, 30)


def test_fit_defaults_validation_to_training():
    """fit() uses x_train/y_train as validation when not provided."""
    reg = _DummyRegressor()
    x = numpy.random.randn(20, 3).astype(numpy.float32)
    y = numpy.random.randn(20).astype(numpy.float32)

    # Should not raise even without x_valid/y_valid
    reg.fit(x, y)
    assert reg.num_channels == 1


def test_loss_history_recorded():
    """Loss history is collected from models that have it."""
    reg = _DummyRegressor()
    x = numpy.random.randn(20, 3).astype(numpy.float32)
    y = numpy.random.randn(20).astype(numpy.float32)

    reg.fit(x, y)
    assert len(reg.loss_history) == 1


def test_fit_resets_stop_flag():
    """fit() resets _stop_fit flag at the beginning."""
    reg = _DummyRegressor()
    reg.stop_fit()
    assert reg._stop_fit is True

    x = numpy.random.randn(20, 3).astype(numpy.float32)
    y = numpy.random.randn(20).astype(numpy.float32)
    reg.fit(x, y)

    assert reg._stop_fit is False


def test_save_load_roundtrip(tmp_path):
    """save/load roundtrip preserves num_channels and prediction consistency."""
    reg = _DummyRegressor()
    x = numpy.random.randn(20, 3).astype(numpy.float32)
    y = numpy.random.randn(20).astype(numpy.float32)
    reg.fit(x, y)

    predictions_before = reg.predict(x)

    save_path = str(tmp_path / 'dummy_regressor')
    reg.save(save_path)

    loaded = RegressorBase.load(save_path)
    assert loaded.num_channels == 1

    predictions_after = loaded.predict(x)
    numpy.testing.assert_array_almost_equal(predictions_before, predictions_after)


def test_predict_before_fit_raises():
    """predict() before fit() raises because models list is empty."""
    reg = _DummyRegressor()
    x = numpy.random.randn(10, 3).astype(numpy.float32)
    with pytest.raises(Exception):
        reg.predict(x)
