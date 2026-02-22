"""Tests for ActivityWidget."""

import pytest

from aydin.gui._qt.custom_widgets.activity_widget import ActivityWidget

pytestmark = pytest.mark.gui


@pytest.fixture
def activity_widget(qtbot, mock_main_page):
    """Create an ActivityWidget."""
    widget = ActivityWidget(mock_main_page)
    qtbot.addWidget(widget)
    yield widget


class TestActivityWidgetInit:
    """Tests for initial widget state."""

    def test_child_widgets_exist(self, activity_widget):
        assert activity_widget.infoTextBox is not None
        assert activity_widget.info_copy_button is not None
        assert activity_widget.info_clear_button is not None
        assert activity_widget.info_save_button is not None
        assert activity_widget.autoscroll_checkbox is not None

    def test_autoscroll_checked_by_default(self, activity_widget):
        assert activity_widget.autoscroll_checkbox.isChecked()

    def test_monospace_font(self, activity_widget):
        from qtpy.QtGui import QFont

        font = activity_widget.infoTextBox.font()
        assert font.styleHint() == QFont.StyleHint.Monospace


class TestActivityPrint:
    """Tests for activity_print method."""

    def test_text_appended(self, activity_widget):
        activity_widget.activity_print("Hello World")
        text = activity_widget.infoTextBox.toPlainText()
        assert "Hello World" in text

    def test_multiple_prints_accumulate(self, activity_widget):
        activity_widget.activity_print("Line 1\n")
        activity_widget.activity_print("Line 2\n")
        text = activity_widget.infoTextBox.toPlainText()
        assert "Line 1" in text
        assert "Line 2" in text


class TestClearActivity:
    """Tests for clear_activity method."""

    def test_clear_removes_text(self, activity_widget):
        activity_widget.activity_print("Some log text")
        activity_widget.clear_activity()
        assert activity_widget.infoTextBox.toPlainText() == ""


class TestCopyLogs:
    """Tests for copy_logs_to_clipboard."""

    def test_clipboard_contains_log_text(self, qtbot, activity_widget):
        from qtpy import QtGui

        activity_widget.activity_print("clipboard test")
        activity_widget.copy_logs_to_clipboard()
        clipboard_text = QtGui.QGuiApplication.clipboard().text()
        assert "clipboard test" in clipboard_text


class TestAutoscrollToggle:
    """Tests for autoscroll checkbox."""

    def test_uncheck_no_crash(self, qtbot, activity_widget):
        activity_widget.autoscroll_checkbox.setChecked(False)
        # Should not raise

    def test_recheck_no_crash(self, qtbot, activity_widget):
        activity_widget.autoscroll_checkbox.setChecked(False)
        activity_widget.autoscroll_checkbox.setChecked(True)
        # Should not raise
