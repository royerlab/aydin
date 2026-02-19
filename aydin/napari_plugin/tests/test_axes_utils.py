"""Tests for napari axis detection utilities."""

from unittest.mock import MagicMock

import numpy
import pytest

from aydin.napari_plugin._axes_utils import (
    _guess_axes_from_shape,
    _labels_are_generic,
    detect_axes_from_napari_layer,
)

# ------------------------------------------------------------------
# _labels_are_generic
# ------------------------------------------------------------------


class TestLabelsAreGeneric:
    def test_empty_strings(self):
        assert _labels_are_generic(('', '', ''))

    def test_integer_labels(self):
        assert _labels_are_generic(('0', '1', '2'))

    def test_named_labels(self):
        assert not _labels_are_generic(('t', 'z', 'y', 'x'))

    def test_mixed_generic_named(self):
        assert not _labels_are_generic(('0', 'y', 'x'))


# ------------------------------------------------------------------
# _guess_axes_from_shape
# ------------------------------------------------------------------


class TestGuessAxesFromShape:
    def test_2d(self):
        assert _guess_axes_from_shape((512, 512)) == 'YX'

    def test_3d_spatial(self):
        assert _guess_axes_from_shape((64, 256, 256)) == 'ZYX'

    def test_3d_channel(self):
        assert _guess_axes_from_shape((256, 256, 3)) == 'YXC'

    def test_4d_spatial(self):
        assert _guess_axes_from_shape((10, 64, 256, 256)) == 'TZYX'

    def test_4d_channel(self):
        assert _guess_axes_from_shape((64, 256, 256, 4)) == 'ZYXC'

    def test_5d_channel(self):
        assert _guess_axes_from_shape((10, 64, 256, 256, 3)) == 'TZYXC'

    def test_5d_no_channel(self):
        assert _guess_axes_from_shape((2, 10, 64, 256, 256)) == 'QTZYX'

    def test_6d(self):
        assert _guess_axes_from_shape((2, 3, 10, 64, 256, 256)) == 'QQTZYX'


# ------------------------------------------------------------------
# detect_axes_from_napari_layer
# ------------------------------------------------------------------


def _make_layer(shape, dtype='float32'):
    """Create a mock napari Image layer."""
    layer = MagicMock()
    layer.data = numpy.zeros(shape, dtype=dtype)
    layer.name = 'test_image'
    return layer


def _make_viewer(axis_labels):
    """Create a mock napari viewer with given axis labels."""
    viewer = MagicMock()
    viewer.dims.axis_labels = axis_labels
    return viewer


class TestDetectAxesFromNapariLayer:
    def test_2d_no_viewer(self):
        layer = _make_layer((256, 256))
        md = detect_axes_from_napari_layer(layer)
        assert md.axes == 'YX'
        assert md.shape == (256, 256)
        assert md.batch_axes == (False, False)
        assert md.channel_axes == (False, False)

    def test_3d_no_viewer(self):
        layer = _make_layer((64, 256, 256))
        md = detect_axes_from_napari_layer(layer)
        assert md.axes == 'ZYX'

    def test_with_named_labels(self):
        layer = _make_layer((10, 64, 256, 256))
        viewer = _make_viewer(('time', 'z', 'y', 'x'))
        md = detect_axes_from_napari_layer(layer, viewer)
        assert md.axes == 'TZYX'

    def test_with_channel_label(self):
        layer = _make_layer((256, 256, 3))
        viewer = _make_viewer(('y', 'x', 'channel'))
        md = detect_axes_from_napari_layer(layer, viewer)
        assert md.axes == 'YXC'
        assert md.channel_axes == (False, False, True)

    def test_generic_labels_fallback(self):
        layer = _make_layer((10, 64, 256, 256))
        viewer = _make_viewer(('0', '1', '2', '3'))
        md = detect_axes_from_napari_layer(layer, viewer)
        assert md.axes == 'TZYX'

    def test_unknown_label_maps_to_Q(self):
        layer = _make_layer((5, 256, 256))
        viewer = _make_viewer(('wavelength', 'y', 'x'))
        md = detect_axes_from_napari_layer(layer, viewer)
        assert md.axes == 'QYX'

    def test_batch_axes_detected(self):
        layer = _make_layer((3, 10, 256, 256))
        viewer = _make_viewer(('wavelength', 't', 'y', 'x'))
        md = detect_axes_from_napari_layer(layer, viewer)
        # 'wavelength' is unknown -> Q; Q is in 'XYZTQC' so not a batch axis
        assert md.axes == 'QTYX'

    def test_dtype_preserved(self):
        layer = _make_layer((256, 256), dtype='uint16')
        md = detect_axes_from_napari_layer(layer)
        assert md.dtype == numpy.dtype('uint16')

    def test_format_is_napari(self):
        layer = _make_layer((256, 256))
        md = detect_axes_from_napari_layer(layer)
        assert md.format == 'napari'
