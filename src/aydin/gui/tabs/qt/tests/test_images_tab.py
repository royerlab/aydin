"""Tests for ImagesTab widget."""

from pathlib import Path

import numpy
import pytest
from qtpy.QtCore import Qt

from aydin.gui.conftest import make_file_metadata
from aydin.gui.tabs.data_model import ImageRecord
from aydin.gui.tabs.qt.images import ImagesTab

pytestmark = pytest.mark.gui


def _populate_images(data_model, n=2, denoise=True):
    """Add synthetic image entries directly to the data model.

    Returns list of entries added.
    """
    entries = []
    for i in range(n):
        path = f"/tmp/synthetic/image_{i}.tif"
        arr = numpy.random.rand(64, 64).astype(numpy.float32)
        meta = make_file_metadata('YX', (64, 64))
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
        entries.append((path, arr, meta))
    return entries


class TestImagesTabInit:
    """Tests for initial ImagesTab state."""

    def test_tree_widget_empty(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.image_list_tree_widget.topLevelItemCount() == 0

    def test_tree_headers(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        header = tab.image_list_tree_widget.headerItem()
        labels = [header.text(i) for i in range(header.columnCount())]
        assert labels == [
            'file name',
            'denoise',
            'axes',
            'shape',
            'dtype',
            'size',
            'output folder',
        ]


class TestImagesTabDataModel:
    """Tests for data model update integration."""

    def test_on_data_model_update_populates_tree(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_images(mock_main_page.data_model, n=3)
        tab.on_data_model_update()
        assert tab.image_list_tree_widget.topLevelItemCount() == 3

    def test_denoise_checkbox_checked_by_default(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_images(mock_main_page.data_model, n=1, denoise=True)
        tab.on_data_model_update()
        item = tab.image_list_tree_widget.topLevelItem(0)
        assert item.checkState(1) == Qt.Checked

    def test_denoise_checkbox_unchecked(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_images(mock_main_page.data_model, n=1, denoise=False)
        tab.on_data_model_update()
        item = tab.image_list_tree_widget.topLevelItem(0)
        assert item.checkState(1) == Qt.Unchecked

    def test_images2denoise_property(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_images(mock_main_page.data_model, n=2, denoise=True)
        tab.on_data_model_update()
        result = tab.images2denoise
        assert result == [True, True]

    def test_images2denoise_mixed(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        # Add one with denoise=True, one with denoise=False
        path1 = "/tmp/synthetic/img_a.tif"
        arr1 = numpy.random.rand(64, 64).astype(numpy.float32)
        meta1 = make_file_metadata('YX', (64, 64))
        mock_main_page.data_model._images.append(
            ImageRecord(
                filename=Path(path1).name,
                array=arr1,
                metadata=meta1,
                denoise=True,
                filepath=path1,
                output_folder=str(Path(path1).parent),
            )
        )
        path2 = "/tmp/synthetic/img_b.tif"
        arr2 = numpy.random.rand(64, 64).astype(numpy.float32)
        meta2 = make_file_metadata('YX', (64, 64))
        mock_main_page.data_model._images.append(
            ImageRecord(
                filename=Path(path2).name,
                array=arr2,
                metadata=meta2,
                denoise=False,
                filepath=path2,
                output_folder=str(Path(path2).parent),
            )
        )
        tab.on_data_model_update()
        result = tab.images2denoise
        assert result == [True, False]


class TestImagesTabRemove:
    """Tests for remove_items_from_tree."""

    def test_remove_clears_tree(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_images(mock_main_page.data_model, n=2)
        tab.on_data_model_update()
        assert tab.image_list_tree_widget.topLevelItemCount() == 2

        tab.remove_items_from_tree()
        assert tab.image_list_tree_widget.topLevelItemCount() == 0

    def test_empty_data_model_clears_tree(self, qtbot, mock_main_page):
        tab = ImagesTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.on_data_model_update()
        assert tab.image_list_tree_widget.topLevelItemCount() == 0
