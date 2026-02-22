"""Tests for FilesTab widget."""

from pathlib import Path

import numpy
import pytest
from qtpy.QtCore import Qt

from aydin.gui.conftest import make_file_metadata
from aydin.gui.tabs.qt.files import FilesTab

pytestmark = pytest.mark.gui


def _populate_data_model(data_model, n=2):
    """Add synthetic file entries directly to the data model internals.

    Returns list of (path, array, metadata) tuples added.
    """
    entries = []
    for i in range(n):
        path = f"/tmp/synthetic/image_{i}.tif"
        arr = numpy.random.rand(64, 64).astype(numpy.float32)
        meta = make_file_metadata('YX', (64, 64))
        data_model._filepaths[path] = (arr, meta)
        data_model._images.append(
            [Path(path).name, arr, meta, True, path, str(Path(path).parent)]
        )
        entries.append((path, arr, meta))
    return entries


class TestFilesTabInit:
    """Tests for initial FilesTab state."""

    def test_tree_widget_empty(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.file_list_tree_widget.topLevelItemCount() == 0

    def test_tree_headers(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        header = tab.file_list_tree_widget.headerItem()
        labels = [header.text(i) for i in range(header.columnCount())]
        assert labels == [
            'File Name',
            'Split Channels',
            'axes',
            'shape',
            'dtype',
            'Path',
        ]

    def test_hyperstack_checkbox_exists(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        assert tab.hyperstack_checkbox is not None
        assert not tab.hyperstack_checkbox.isChecked()


class TestFilesTabDataModel:
    """Tests for data model update integration."""

    def test_on_data_model_update_populates_tree(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=3)
        tab.on_data_model_update()
        assert tab.file_list_tree_widget.topLevelItemCount() == 3

    def test_on_data_model_update_shows_filename(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=1)
        tab.on_data_model_update()
        item = tab.file_list_tree_widget.topLevelItem(0)
        assert item.text(0) == "image_0.tif"

    def test_on_data_model_update_shows_axes(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=1)
        tab.on_data_model_update()
        item = tab.file_list_tree_widget.topLevelItem(0)
        assert item.text(2) == "YX"

    def test_on_data_model_update_shows_path(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=1)
        tab.on_data_model_update()
        item = tab.file_list_tree_widget.topLevelItem(0)
        assert item.text(5) == "/tmp/synthetic/image_0.tif"

    def test_split_channels_checkbox_unchecked(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=1)
        tab.on_data_model_update()
        item = tab.file_list_tree_widget.topLevelItem(0)
        assert item.checkState(1) == Qt.Unchecked


class TestFilesTabRemoveAll:
    """Tests for remove_all_items_from_tree."""

    def test_remove_all_clears_tree(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        _populate_data_model(mock_main_page.data_model, n=2)
        tab.on_data_model_update()
        assert tab.file_list_tree_widget.topLevelItemCount() == 2

        tab.remove_all_items_from_tree()
        assert tab.file_list_tree_widget.topLevelItemCount() == 0

    def test_empty_data_model_shows_nothing(self, qtbot, mock_main_page):
        tab = FilesTab(mock_main_page)
        qtbot.addWidget(tab)
        tab.on_data_model_update()
        assert tab.file_list_tree_widget.topLevelItemCount() == 0
