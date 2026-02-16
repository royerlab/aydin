"""Tests for DimensionsTab widget."""

from pathlib import Path

import numpy
import pytest
from qtpy.QtCore import Qt

from aydin.gui.conftest import make_file_metadata
from aydin.gui.tabs.data_model import ImageRecord
from aydin.gui.tabs.qt.dimensions import DimensionsTab

pytestmark = pytest.mark.gui


def _add_single_image(data_model, axes, shape, denoise=True):
    """Add a single synthetic image to data_model._images for dimension tests."""
    path = "/tmp/synthetic/test_image.tif"
    arr = numpy.random.rand(*shape).astype(numpy.float32)
    meta = make_file_metadata(axes, shape)
    data_model._filepaths[path] = (arr, meta)
    data_model._images.append(
        ImageRecord(
            filename=Path(path).name,
            array=arr,
            metadata=meta,
            denoise=denoise,
            filepath=path,
            output_folder=str(Path(path).parent),
        )
    )
    return meta


def _get_size_row_values(tab, n_axes):
    """Read the size row values from the dimensions tree for n_axes columns."""
    root = tab.dimensions_tree.invisibleRootItem()
    if not root.childCount():
        return None
    size_row = root.child(0)
    return [int(size_row.text(x)) for x in range(1, n_axes + 1)]


class TestDimensionsTabInit:
    """Tests for initial DimensionsTab state."""

    def test_empty_state(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.dimensions is None
        assert tab.axes is None

    def test_spatiotemporal_none_when_empty(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.spatiotemporal_axes is None

    def test_batch_none_when_empty(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.batch_axes is None


class TestDimensionsTab2D:
    """Tests for 2D YX metadata."""

    def test_axes_yx(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        assert tab.axes == 'YX'

    def test_dimensions_populated(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        dims = _get_size_row_values(tab, 2)
        assert dims == [512, 512]

    def test_spatiotemporal_all_true(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        assert tab.spatiotemporal_axes == [True, True]

    def test_batch_all_false(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        assert tab.batch_axes == [False, False]


class TestDimensionsTab3D:
    """Tests for 3D ZYX metadata."""

    def test_axes_zyx(self, qtbot, mock_main_page, sample_metadata_3d):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_3d
        assert tab.axes == 'ZYX'

    def test_dimensions_3d(self, qtbot, mock_main_page, sample_metadata_3d):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_3d
        dims = _get_size_row_values(tab, 3)
        assert dims == [32, 256, 256]

    def test_spatiotemporal_3d(self, qtbot, mock_main_page, sample_metadata_3d):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_3d
        assert tab.spatiotemporal_axes == [True, True, True]


class TestDimensionsTab4D:
    """Tests for 4D TZYX metadata with special case handling."""

    def test_axes_tzyx(self, qtbot, mock_main_page, sample_metadata_4d):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_4d
        assert tab.axes == 'TZYX'

    def test_dimensions_4d(self, qtbot, mock_main_page, sample_metadata_4d):
        """The dimensions getter works correctly for 4D data."""
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_4d
        dims = tab.dimensions
        assert dims == [10, 32, 256, 256]

    def test_t_auto_assigned_as_batch(self, qtbot, mock_main_page, sample_metadata_4d):
        """handle_special_cases() should auto-assign T as batch for 4D TZYX."""
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_4d
        # T is at index 0; should be batch
        assert tab.batch_axes[0] is True
        # T should not be spatiotemporal
        assert tab.spatiotemporal_axes[0] is False

    def test_spatial_axes_remain_spatiotemporal(
        self, qtbot, mock_main_page, sample_metadata_4d
    ):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata_4d
        # Z, Y, X should remain spatiotemporal
        assert tab.spatiotemporal_axes[1:] == [True, True, True]


class TestDimensionsTabClear:
    """Tests for setting dimensions to None."""

    def test_setting_none_clears(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        # Verify tree has children after setting
        root = tab.dimensions_tree.invisibleRootItem()
        assert root.childCount() > 0
        tab.dimensions = None
        assert tab.dimensions is None

    def test_setting_none_clears_tree(self, qtbot, mock_main_page, sample_metadata):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = sample_metadata
        tab.dimensions = None
        root = tab.dimensions_tree.invisibleRootItem()
        assert root.childCount() == 0


class TestDimensionsTabExclusiveStates:
    """Tests for on_tree_clicked mutual exclusivity logic."""

    def _make_3d_tab(self, qtbot, mock_main_page):
        """Create a DimensionsTab loaded with 3D ZYX metadata."""
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        meta = make_file_metadata('ZYX', (16, 64, 64))
        tab.dimensions = meta
        return tab

    def test_checking_batch_unchecks_spatiotemporal(self, qtbot, mock_main_page):
        """Checking batch for an axis should uncheck spatiotemporal for that axis."""
        tab = self._make_3d_tab(qtbot, mock_main_page)
        # Z is column 1. Initially spatiotemporal=checked, batch=unchecked
        assert tab.spatiotemporal_axes[0] is True
        assert tab.batch_axes[0] is False
        # Simulate checking batch for Z
        tab.batch_row.setCheckState(1, Qt.Checked)
        tab.on_tree_clicked(tab.batch_row, 1)
        assert tab.spatiotemporal_axes[0] is False

    def test_checking_channel_unchecks_spatiotemporal(self, qtbot, mock_main_page):
        """Checking channel for an axis should uncheck spatiotemporal."""
        tab = self._make_3d_tab(qtbot, mock_main_page)
        tab.channel_row.setCheckState(1, Qt.Checked)
        tab.on_tree_clicked(tab.channel_row, 1)
        assert tab.spatiotemporal_axes[0] is False
        assert tab.channel_axes[0] is True

    def test_cannot_uncheck_last_spatiotemporal_via_batch(self, qtbot, mock_main_page):
        """Cannot remove the last spatiotemporal dimension by checking batch."""
        tab = self._make_3d_tab(qtbot, mock_main_page)
        # Make Z and Y batch, leaving only X as spatiotemporal
        tab.batch_row.setCheckState(1, Qt.Checked)
        tab.on_tree_clicked(tab.batch_row, 1)
        tab.batch_row.setCheckState(2, Qt.Checked)
        tab.on_tree_clicked(tab.batch_row, 2)
        assert sum(int(x) for x in tab.spatiotemporal_axes) == 1
        # Try to check batch for X (the last spatiotemporal) — should be rejected
        tab.batch_row.setCheckState(3, Qt.Checked)
        tab.on_tree_clicked(tab.batch_row, 3)
        # X should still be spatiotemporal
        assert tab.spatiotemporal_axes[2] is True
        assert tab.batch_axes[2] is False

    def test_unchecking_only_checked_row_rejects(self, qtbot, mock_main_page):
        """Unchecking the only checked row for a column re-checks it."""
        tab = self._make_3d_tab(qtbot, mock_main_page)
        # Z is spatiotemporal. Uncheck spatiotemporal for Z — since batch
        # and channel are unchecked too, it should re-check itself.
        tab.spatiotemporal_row.setCheckState(1, Qt.Unchecked)
        tab.on_tree_clicked(tab.spatiotemporal_row, 1)
        assert tab.spatiotemporal_axes[0] is True

    def test_checking_spatiotemporal_unchecks_batch(self, qtbot, mock_main_page):
        """Checking spatiotemporal should uncheck batch for that axis."""
        tab = self._make_3d_tab(qtbot, mock_main_page)
        # First make Z a batch dimension
        tab.batch_row.setCheckState(1, Qt.Checked)
        tab.on_tree_clicked(tab.batch_row, 1)
        assert tab.batch_axes[0] is True
        assert tab.spatiotemporal_axes[0] is False
        # Now check spatiotemporal for Z
        tab.spatiotemporal_row.setCheckState(1, Qt.Checked)
        tab.on_tree_clicked(tab.spatiotemporal_row, 1)
        assert tab.spatiotemporal_axes[0] is True
        assert tab.batch_axes[0] is False


class TestDimensionsTabChannelAxes:
    """Tests for channel axis handling."""

    def test_channel_axes_all_false_by_default(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        meta = make_file_metadata('ZYX', (16, 64, 64))
        tab.dimensions = meta
        assert tab.channel_axes == [False, False, False]

    def test_channel_axes_from_metadata(self, qtbot, mock_main_page):
        """Channel axis set in metadata should be reflected after init."""
        meta = make_file_metadata('CYX', (3, 64, 64))
        meta.channel_axes = (True, False, False)
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.dimensions = meta
        assert tab.channel_axes[0] is True
        assert tab.spatiotemporal_axes[0] is False


class TestDimensionsTabDataModel:
    """Tests for on_data_model_update integration."""

    def test_single_image_populates(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        _add_single_image(mock_main_page.data_model, 'ZYX', (16, 64, 64))
        tab.on_data_model_update()
        assert tab.axes == 'ZYX'

    def test_no_images_clears(self, qtbot, mock_main_page):
        tab = DimensionsTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.on_data_model_update()
        assert tab.dimensions is None
