"""Tests for the Aydin Denoiser napari dock widget."""

from unittest.mock import MagicMock

import numpy
import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def mock_viewer():
    """Create a minimal mock napari viewer."""
    viewer = MagicMock()
    viewer.dims.axis_labels = ('y', 'x')

    # Provide layer events
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()

    # Start with no layers
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))
    viewer.layers.selection = []

    return viewer


@pytest.fixture
def widget(qtbot, mock_viewer):
    """Create an AydinDenoiseWidget attached to a mock viewer."""
    from aydin.napari_plugin._widget import AydinDenoiseWidget

    w = AydinDenoiseWidget(mock_viewer)
    qtbot.addWidget(w)
    return w


class TestWidgetInstantiation:
    def test_creates_widget(self, widget):
        assert widget is not None

    def test_has_denoise_button(self, widget):
        assert widget._denoise_btn is not None
        assert widget._denoise_btn.text() == 'Denoise'

    def test_has_cancel_button(self, widget):
        assert widget._cancel_btn is not None
        assert not widget._cancel_btn.isEnabled()

    def test_has_studio_button(self, widget):
        assert widget._studio_btn is not None
        assert widget._studio_btn.text() == 'Open Aydin Studio'


class TestLayerCombo:
    def test_empty_when_no_layers(self, widget):
        assert widget._layer_combo.count() == 0

    def test_updates_on_add(self, qtbot, mock_viewer):
        """Adding an Image layer triggers combo refresh."""
        import napari.layers

        layer = MagicMock(spec=napari.layers.Image)
        layer.name = 'test_img'
        layer.data = numpy.zeros((64, 64), dtype='float32')

        mock_viewer.layers.__iter__ = MagicMock(return_value=iter([layer]))

        from aydin.napari_plugin._widget import AydinDenoiseWidget

        w = AydinDenoiseWidget(mock_viewer)
        qtbot.addWidget(w)

        assert w._layer_combo.count() == 1
        assert w._layer_combo.itemText(0) == 'test_img'


class TestMethodCombo:
    def test_methods_populated(self, widget):
        assert widget._method_combo.count() == 8

    def test_first_method_is_fgr_cb(self, widget):
        assert 'FGR CatBoost' in widget._method_combo.itemText(0)

    def test_methods_have_tooltips(self, widget):
        from qtpy.QtCore import Qt

        for i in range(widget._method_combo.count()):
            tooltip = widget._method_combo.itemData(i, Qt.ToolTipRole)
            assert tooltip and len(tooltip) > 10


class TestDimensionDisplay:
    def test_no_layer_shows_placeholder(self, widget):
        assert 'select a layer' in widget._axes_label.text()

    def test_layer_shows_axes(self, qtbot, mock_viewer):
        import napari.layers

        layer = MagicMock(spec=napari.layers.Image)
        layer.name = 'img_3d'
        layer.data = numpy.zeros((32, 64, 64), dtype='float32')

        mock_viewer.layers.__iter__ = MagicMock(return_value=iter([layer]))
        mock_viewer.dims.axis_labels = ('z', 'y', 'x')

        from aydin.napari_plugin._widget import AydinDenoiseWidget

        w = AydinDenoiseWidget(mock_viewer)
        qtbot.addWidget(w)

        assert 'ZYX' in w._axes_label.text()
