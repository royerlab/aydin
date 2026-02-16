"""Tests for CatBoost training callbacks."""

from unittest.mock import Mock

from aydin.regression.cb_utils.callbacks import CatBoostStopTrainingCallback


def test_callback_initial_state():
    """Callback starts with continue_training=True."""
    cb = CatBoostStopTrainingCallback()
    assert cb.continue_training is True


def test_callback_returns_true_by_default():
    """after_iteration returns True when continue_training is True."""
    cb = CatBoostStopTrainingCallback()
    info = Mock()
    info.iteration = 0
    info.metrics = {'learn': {'MAE': [0.5]}, 'test': {'MAE': [0.6]}}

    result = cb.after_iteration(info)
    assert result is True


def test_callback_returns_false_when_stopped():
    """after_iteration returns False when continue_training is set to False."""
    cb = CatBoostStopTrainingCallback()
    cb.continue_training = False

    info = Mock()
    info.iteration = 10
    info.metrics = {'learn': {'MAE': [0.3]}, 'test': {'MAE': [0.4]}}

    result = cb.after_iteration(info)
    assert result is False


def test_callback_reads_metrics():
    """after_iteration accesses iteration and metrics without error."""
    cb = CatBoostStopTrainingCallback()
    info = Mock()
    info.iteration = 42
    info.metrics = {'learn': {'MAE': [0.1, 0.05]}, 'test': {'MAE': [0.2, 0.15]}}

    result = cb.after_iteration(info)
    assert result is True
