"""Tests for Overlay loading indicator widget."""

import pytest

from aydin.gui._qt.custom_widgets.overlay import Overlay

pytestmark = pytest.mark.gui


class TestOverlayInit:
    """Tests for initial construction."""

    def test_default_construction(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        assert overlay.timer is None
        assert overlay.counter == 0

    def test_starts_hidden(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        assert not overlay.isVisible()


class TestOverlayShowHide:
    """Tests for show/hide timer lifecycle."""

    def test_show_starts_timer(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.show()
        assert overlay.timer is not None
        assert overlay.counter == 0
        overlay.hide()

    def test_hide_stops_timer(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.show()
        assert overlay.timer is not None
        overlay.hide()
        assert overlay.timer is None

    def test_show_hide_show_cycle(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.show()
        timer1 = overlay.timer
        overlay.hide()
        assert overlay.timer is None
        overlay.show()
        assert overlay.timer is not None
        # New timer should be different from old one
        assert overlay.timer != timer1
        overlay.hide()

    def test_hide_without_show_no_crash(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        # timer is None, hideEvent should handle gracefully
        overlay.hide()
        assert overlay.timer is None


class TestOverlayAnimation:
    """Tests for animation counter."""

    def test_timer_event_increments_counter(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.show()
        initial = overlay.counter
        # Simulate timer events
        from qtpy.QtCore import QTimerEvent

        overlay.timerEvent(QTimerEvent(overlay.timer))
        assert overlay.counter == initial + 1
        overlay.timerEvent(QTimerEvent(overlay.timer))
        assert overlay.counter == initial + 2
        overlay.hide()

    def test_show_resets_counter(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.show()
        overlay.counter = 42
        overlay.hide()
        overlay.show()
        assert overlay.counter == 0
        overlay.hide()


class TestOverlayPaint:
    """Tests for paint event handling."""

    def test_paint_does_not_crash(self, qtbot):
        overlay = Overlay()
        qtbot.addWidget(overlay)
        overlay.resize(200, 200)
        overlay.show()
        # Force a repaint — should not crash
        overlay.repaint()
        overlay.hide()
