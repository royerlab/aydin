"""Tests for QRangeSliderWithLabels widget."""

from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QWidget

from aydin.gui._qt.custom_widgets.range_slider_with_labels import QRangeSliderWithLabels

pytestmark = pytest.mark.gui


@pytest.fixture
def mock_parent(qtbot):
    """Create a mock parent with the methods RangeSliderWithLabels calls."""
    parent = QWidget()
    parent.update_current_viewer_dims = Mock()
    parent.update_crop_label_layer = Mock()
    parent.update_summary = Mock()
    qtbot.addWidget(parent)
    return parent


class TestRangeSliderWithLabelsInit:
    """Tests for initial construction."""

    def test_default_label(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent)
        qtbot.addWidget(slider)
        assert slider.slider_label.text() == "N/A"

    def test_custom_label(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="X")
        qtbot.addWidget(slider)
        assert slider.slider_label.text() == "X"

    def test_default_size(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, size=200)
        qtbot.addWidget(slider)
        assert slider.size == 200
        assert slider.slider.range() == (0, 200)

    def test_initial_cutoffs(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, size=100)
        qtbot.addWidget(slider)
        assert slider.lower_cutoff == 0
        assert slider.upper_cutoff == 100

    def test_range_label_text(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, size=256)
        qtbot.addWidget(slider)
        assert slider.range_label.text() == "[0,256)"

    def test_select_all_button_exists(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent)
        qtbot.addWidget(slider)
        assert slider.select_all_button.text() == "Select All"

    def test_limit_labels_disabled(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent)
        qtbot.addWidget(slider)
        assert not slider.lower_limit_label.isEnabled()
        assert not slider.upper_limit_label.isEnabled()


class TestRangeSliderWithLabelsValues:
    """Tests for value changes and label updates."""

    def test_slider_value_change_updates_labels(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Z", size=100)
        qtbot.addWidget(slider)
        slider.slider.set_values((10, 90))
        assert slider.lower_cutoff == 10
        assert slider.upper_cutoff == 90

    def test_slider_value_change_calls_parent(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Z", size=100)
        qtbot.addWidget(slider)
        slider.slider.set_values((10, 90))
        mock_parent.update_crop_label_layer.assert_called()
        mock_parent.update_summary.assert_called()

    def test_non_xy_calls_update_viewer_dims(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Z", size=100)
        qtbot.addWidget(slider)
        slider.slider.set_values((5, 80))
        mock_parent.update_current_viewer_dims.assert_called()


class TestRangeSliderWithLabelsMinLength:
    """Tests for minimum range length enforcement on X/Y axes."""

    def test_x_slider_enforces_min_length(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="X", size=100, min_length=32)
        qtbot.addWidget(slider)
        # Try to set range smaller than min_length
        slider.slider.set_values((40, 50))
        # Should have been rejected — cutoffs should stay at original values
        assert slider.upper_cutoff - slider.lower_cutoff >= 32

    def test_y_slider_enforces_min_length(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Y", size=100, min_length=32)
        qtbot.addWidget(slider)
        slider.slider.set_values((40, 50))
        assert slider.upper_cutoff - slider.lower_cutoff >= 32

    def test_z_slider_allows_small_range(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Z", size=100, min_length=32)
        qtbot.addWidget(slider)
        slider.slider.set_values((40, 50))
        # Z is not X or Y, so min_length is not enforced
        assert slider.lower_cutoff == 40
        assert slider.upper_cutoff == 50


class TestRangeSliderWithLabelsSelectAll:
    """Tests for the Select All button."""

    def test_select_all_resets_to_full_range(self, qtbot, mock_parent):
        slider = QRangeSliderWithLabels(mock_parent, label="Z", size=100)
        qtbot.addWidget(slider)
        slider.slider.set_values((20, 80))
        assert slider.lower_cutoff == 20
        slider.select_all_button.click()
        assert slider.lower_cutoff == 0
        assert slider.upper_cutoff == 100
