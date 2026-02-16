"""Tests for QRangeSlider and QHRangeSlider widgets."""

import pytest
from qtpy.QtGui import QColor

from aydin.gui._qt.custom_widgets.range_slider import QHRangeSlider

pytestmark = pytest.mark.gui


class TestQHRangeSliderDefaults:
    """Tests for default construction of QHRangeSlider."""

    def test_default_values(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        assert slider.values() == (20, 80)

    def test_default_range(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        assert slider.range() == (0, 100)

    def test_default_not_collapsed(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        assert slider.collapsed is False
        assert slider.collapsible is True


class TestQHRangeSliderCustom:
    """Tests for custom construction parameters."""

    def test_custom_initial_values(self, qtbot):
        slider = QHRangeSlider(initial_values=(10, 50))
        qtbot.addWidget(slider)
        assert slider.values() == (10, 50)

    def test_custom_data_range(self, qtbot):
        slider = QHRangeSlider(data_range=(0, 200))
        qtbot.addWidget(slider)
        assert slider.range() == (0, 200)

    def test_custom_step_size(self, qtbot):
        slider = QHRangeSlider(data_range=(0, 100), step_size=5)
        qtbot.addWidget(slider)
        assert slider._step == 5


class TestQHRangeSliderSignals:
    """Tests for signal emission."""

    def test_set_values_emits_values_changed(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        with qtbot.waitSignal(slider.valuesChanged, timeout=1000):
            slider.set_values((30, 70))

    def test_set_range_emits_range_changed(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        with qtbot.waitSignal(slider.rangeChanged, timeout=1000):
            slider.set_range((0, 200))


class TestQHRangeSliderCollapseExpand:
    """Tests for collapse/expand behavior."""

    def test_collapse(self, qtbot):
        slider = QHRangeSlider(initial_values=(20, 80))
        qtbot.addWidget(slider)
        slider.collapse()
        assert slider.collapsed is True
        vals = slider.slider_values()
        assert vals[0] == vals[1]  # min == max when collapsed

    def test_expand_after_collapse(self, qtbot):
        slider = QHRangeSlider(initial_values=(20, 80))
        qtbot.addWidget(slider)
        original_values = slider.slider_values()
        slider.collapse()
        slider.expand()
        assert slider.collapsed is False
        restored = slider.slider_values()
        # Values should be approximately restored
        assert (
            abs(restored[1] - restored[0] - (original_values[1] - original_values[0]))
            < 0.01
        )


class TestQHRangeSliderColors:
    """Tests for color property get/set."""

    def test_bar_color(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        color = QColor(255, 0, 0)
        slider.setBarColor(color)
        assert slider.getBarColor() == color

    def test_handle_color(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        color = QColor(0, 255, 0)
        slider.setHandleColor(color)
        assert slider.getHandleColor() == color

    def test_handle_border_color(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        color = QColor(0, 0, 255)
        slider.setHandleBorderColor(color)
        assert slider.getHandleBorderColor() == color

    def test_background_color(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        color = QColor(128, 128, 128)
        slider.setBackgroundColor(color)
        assert slider.getBackgroundColor() == color


class TestQHRangeSliderEnabled:
    """Tests for enabled/disabled state."""

    def test_disable_and_enable(self, qtbot):
        slider = QHRangeSlider()
        qtbot.addWidget(slider)
        slider.setEnabled(False)
        assert not slider.isEnabled()
        slider.setEnabled(True)
        assert slider.isEnabled()
