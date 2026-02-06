"""Tests for Aydin logging utilities.

Tests cover:
- Basic lprint/lsection functionality
- Depth tracking
- Test suppression behavior
- GUI callback emission
- ANSI stripping for GUI
- Configuration options (max_depth, elapsed_time)
"""

import io
import sys
from unittest.mock import MagicMock

import pytest

from aydin.util.log.log import Log, aprint, asection, lprint, lsection


class TestBasicLogging:
    """Test basic lprint and lsection functionality."""

    def test_lprint_basic(self, capsys):
        """Test that lprint outputs correctly when enabled."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        lprint('Hello world')

        captured = capsys.readouterr()
        assert 'Hello world' in captured.out

        Log.override_test_exclusion = False
        Log.enable_output = False

    def test_lprint_with_multiple_args(self, capsys):
        """Test lprint with multiple arguments."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        lprint('a', 'b', 'c')

        captured = capsys.readouterr()
        assert 'a b c' in captured.out

        Log.override_test_exclusion = False
        Log.enable_output = False

    def test_lprint_suppressed_during_tests(self, capsys):
        """Test that lprint is suppressed during test runs by default."""
        # Ensure override is off
        Log.override_test_exclusion = False
        Log.enable_output = True

        lprint('Should not appear')

        captured = capsys.readouterr()
        assert 'Should not appear' not in captured.out

        Log.enable_output = False

    def test_lsection_outputs_header(self, capsys):
        """Test that lsection outputs the section header."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        with lsection('Test Section'):
            pass

        captured = capsys.readouterr()
        assert 'Test Section' in captured.out

        Log.override_test_exclusion = False
        Log.enable_output = False


class TestDepthTracking:
    """Test depth tracking during nested sections."""

    def test_initial_depth_is_zero(self):
        """Test that initial depth is 0."""
        assert Log.depth == 0

    def test_depth_increments_in_section(self):
        """Test that depth increases inside lsection."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        assert Log.depth == 0
        with lsection('Level 1'):
            assert Log.depth == 1
            with lsection('Level 2'):
                assert Log.depth == 2
            assert Log.depth == 1
        assert Log.depth == 0

        Log.override_test_exclusion = False
        Log.enable_output = False

    @pytest.mark.heavy
    def test_deeply_nested_sections(self):
        """Test depth tracking with deeply nested sections."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        with lsection('a section'):
            lprint('a line')

            with lsection('a subsection'):
                lprint('another line')

                with lsection('a subsection'):
                    assert Log.depth == 3

                    with lsection('a subsection'):
                        with lsection('a subsection'):
                            assert Log.depth == 5

                            with lsection('a subsection'):
                                with lsection('a subsection'):
                                    assert Log.depth == 7

        lprint('test is finished...')
        assert Log.depth == 0

        Log.override_test_exclusion = False
        Log.enable_output = False


class TestGUICallback:
    """Test GUI callback emission."""

    def test_gui_callback_emission(self):
        """Test that GUI callback receives log output."""
        Log.override_test_exclusion = True
        Log.enable_output = False  # Disable console output
        Log.guiEnabled = True

        mock_callback = MagicMock()
        Log.gui_callback = mock_callback

        lprint('Test message')

        # Callback should have been called
        mock_callback.emit.assert_called()
        call_args = mock_callback.emit.call_args[0][0]
        assert 'Test message' in call_args

        # Cleanup
        Log.gui_callback = None
        Log.override_test_exclusion = False

    def test_gui_callback_strips_ansi(self):
        """Test that ANSI codes are stripped before GUI emission."""
        Log.override_test_exclusion = True
        Log.enable_output = False
        Log.guiEnabled = True

        mock_callback = MagicMock()
        Log.gui_callback = mock_callback

        # Call native_print directly with ANSI codes
        Log.native_print('\x1b[31mRed Text\x1b[0m')

        call_args = mock_callback.emit.call_args[0][0]
        # Should not contain ANSI escape sequences
        assert '\x1b[' not in call_args
        assert 'Red Text' in call_args

        # Cleanup
        Log.gui_callback = None
        Log.override_test_exclusion = False

    def test_gui_statusbar_update(self):
        """Test that GUI status bar receives updates."""
        Log.override_test_exclusion = True
        Log.enable_output = False
        Log.guiEnabled = True

        mock_callback = MagicMock()
        mock_statusbar = MagicMock()
        Log.gui_callback = mock_callback
        Log.gui_statusbar = mock_statusbar

        lprint('Status message')

        mock_statusbar.showMessage.assert_called()
        call_args = mock_statusbar.showMessage.call_args[0][0]
        assert 'Status message' in call_args

        # Cleanup
        Log.gui_callback = None
        Log.gui_statusbar = None
        Log.override_test_exclusion = False


class TestConfiguration:
    """Test configuration options."""

    def test_max_depth_setting(self):
        """Test set_log_max_depth configuration."""
        Log.set_log_max_depth(3)
        assert Log.max_depth == 2  # Internal is 0-indexed

        # Reset
        Log.max_depth = float('inf')

    def test_elapsed_time_setting(self, capsys):
        """Test set_log_elapsed_time configuration."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        # With elapsed time disabled
        Log.set_log_elapsed_time(False)
        with lsection('No timing'):
            pass

        captured = capsys.readouterr()
        # Should not contain timing indicators
        assert 'microseconds' not in captured.out
        assert 'milliseconds' not in captured.out
        assert 'seconds' not in captured.out

        # Reset
        Log.set_log_elapsed_time(True)
        Log.override_test_exclusion = False
        Log.enable_output = False

    def test_elapsed_time_shown_by_default(self, capsys):
        """Test that elapsed time is shown by default."""
        Log.override_test_exclusion = True
        Log.enable_output = True
        Log.log_elapsed_time = True

        with lsection('With timing'):
            pass

        captured = capsys.readouterr()
        # Should contain some timing indicator
        assert (
            'microseconds' in captured.out
            or 'milliseconds' in captured.out
            or 'seconds' in captured.out
        )

        Log.override_test_exclusion = False
        Log.enable_output = False


class TestTestContext:
    """Test the test_context context manager."""

    def test_context_enables_output(self, capsys):
        """Test that test_context enables output during tests."""
        # Ensure we start with output suppressed
        Log.override_test_exclusion = False
        Log.enable_output = True

        with Log.test_context():
            lprint('Inside test context')

        captured = capsys.readouterr()
        assert 'Inside test context' in captured.out

        Log.enable_output = False


class TestAliases:
    """Test that aprint/asection are aliases for lprint/lsection."""

    def test_aprint_is_lprint(self):
        """Test that aprint is the same as lprint."""
        assert aprint is lprint

    def test_asection_is_lsection(self):
        """Test that asection is the same as lsection."""
        assert asection is lsection


class TestExceptionHandling:
    """Test exception handling in sections."""

    def test_exception_propagates_from_section(self):
        """Test that exceptions inside lsection are propagated."""
        Log.override_test_exclusion = True
        Log.enable_output = True

        with pytest.raises(ValueError, match='test error'):
            with lsection('Error section'):
                raise ValueError('test error')

        # Depth should be reset even after exception
        assert Log.depth == 0

        Log.override_test_exclusion = False
        Log.enable_output = False
