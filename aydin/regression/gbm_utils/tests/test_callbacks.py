"""Tests for LightGBM training callbacks."""

from unittest.mock import Mock

import pytest
from lightgbm.callback import EarlyStopException

from aydin.regression.gbm_utils.callbacks import _format_eval_result, early_stopping


class TestFormatEvalResult:
    """Tests for _format_eval_result helper."""

    def test_four_element_tuple(self):
        """Format a 4-element LightGBM eval result."""
        result = _format_eval_result(('valid_0', 'l1', 0.123456, False))
        assert "valid_0" in result
        assert "l1" in result
        assert "0.123456" in result

    def test_five_element_tuple_with_stdv(self):
        """Format a 5-element result with standard deviation."""
        result = _format_eval_result(
            ('valid_0', 'l1', 0.1, False, 0.05), show_stdv=True
        )
        assert "0.05" in result

    def test_five_element_tuple_no_stdv(self):
        """Format a 5-element result when show_stdv=False."""
        result = _format_eval_result(
            ('valid_0', 'l1', 0.1, False, 0.05), show_stdv=False
        )
        assert "0.05" not in result

    def test_five_element_tuple_empty_stdv(self):
        """Format a 5-element result when stdv element is empty/falsy."""
        result = _format_eval_result(('valid_0', 'l1', 0.1, False, None))
        assert "None" not in result

    def test_unknown_length_fallback(self):
        """Fallback to str() for unexpected tuple lengths."""
        result = _format_eval_result(('a', 'b', 'c'))
        assert isinstance(result, str)


class TestEarlyStopping:
    """Tests for the early_stopping callback."""

    def _make_env(self, iteration=0, end_iteration=100, params=None):
        """Create a mock LightGBM training environment."""
        env = Mock()
        env.iteration = iteration
        env.end_iteration = end_iteration
        env.params = params or {}
        env.evaluation_result_list = [('valid_0', 'l1', 0.5 - iteration * 0.001, False)]
        return env

    def _make_regressor(self):
        """Create a mock regressor with _stop_fit flag."""
        reg = Mock()
        reg._stop_fit = False
        return reg

    def test_early_stopping_init_on_first_call(self):
        """Callback should initialize on first iteration."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=10)
        env = self._make_env(iteration=0)

        callback(env)  # Should not raise

    def test_early_stopping_continues_while_improving(self):
        """Callback should not raise when score keeps improving."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=5)

        for i in range(5):
            env = self._make_env(iteration=i)
            env.evaluation_result_list = [('valid_0', 'l1', 1.0 - i * 0.1, False)]
            callback(env)

    def test_early_stopping_raises_after_patience(self):
        """Callback should raise EarlyStopException after stopping_rounds."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=3)

        # First call with a good score
        env = self._make_env(iteration=0)
        env.evaluation_result_list = [('valid_0', 'l1', 0.1, False)]
        callback(env)

        # Subsequent calls with no improvement
        with pytest.raises(EarlyStopException):
            for i in range(1, 10):
                env = self._make_env(iteration=i)
                env.evaluation_result_list = [('valid_0', 'l1', 0.5, False)]
                callback(env)

    def test_early_stopping_external_stop(self):
        """Callback should raise when regressor._stop_fit is True."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=100)

        # Initialize
        env = self._make_env(iteration=0)
        env.evaluation_result_list = [('valid_0', 'l1', 0.5, False)]
        callback(env)

        # Trigger external stop
        reg._stop_fit = True
        env = self._make_env(iteration=1)
        env.evaluation_result_list = [('valid_0', 'l1', 0.5, False)]
        with pytest.raises(EarlyStopException):
            callback(env)

    def test_early_stopping_disabled_in_dart_mode(self):
        """Early stopping should be disabled in dart boosting mode."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=3)

        env = self._make_env(iteration=0, params={'boosting_type': 'dart'})
        # Should not raise, just warn and disable
        callback(env)

        # Further calls should not raise either
        env = self._make_env(iteration=1, params={'boosting_type': 'dart'})
        callback(env)

    def test_early_stopping_raises_at_end_iteration(self):
        """Callback should raise at end_iteration with best result."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=100)

        # Initialize with a score
        env = self._make_env(iteration=0, end_iteration=3)
        env.evaluation_result_list = [('valid_0', 'l1', 0.5, False)]
        callback(env)

        env = self._make_env(iteration=1, end_iteration=3)
        env.evaluation_result_list = [('valid_0', 'l1', 0.4, False)]
        callback(env)

        # At end_iteration - 1, should raise
        env = self._make_env(iteration=2, end_iteration=3)
        env.evaluation_result_list = [('valid_0', 'l1', 0.45, False)]
        with pytest.raises(EarlyStopException):
            callback(env)

    def test_callback_order(self):
        """Callback should have order=30."""
        reg = self._make_regressor()
        callback = early_stopping(reg, stopping_rounds=5)
        assert callback.order == 30
