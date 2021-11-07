from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
)

from aydin.gui._qt.custom_widgets.readmoreless_label import QReadMoreLessLabel
from aydin.io.io import FileMetadata


class DimensionsTab(QWidget):
    """
    Interpreting image dimensions <br><br> Images can have many dimensions: 2D, 3D, 4D, 3D+t... Some dimensions are
    'spatio-temporal' and the signal is expected to have a degree of continuity and correlation across these
    dimensions. Other dimensions are 'batch' dimensions, they just state that we have multiple images of the same
    kind and shape.

    <moreless> <br><br> Finally, some dimensions are 'channel' dimensions and carry vectorial information for each
    voxel of the image. In this tab, you can help aydin better denoise your images by telling it how to interpret the
    dimensions of your images. The choices made here will impact denoising speed and quality.

    In general, we don't recommend denoising multi-channel images unless the correlation between the channels is
    strong and expected. Denoising each channel separately reduces the chance of channel 'bleed-through'. If you
    expect continuity along certain dimensions do not select them as 'batch' dimensions.

    <split>
    In general, denoising leverages signal continuity and correlation and thus would benefit from as many 'spatio-temporal'
    dimensions as possible. However, the more spatio-temporal dimensions, the more intense the computation,
    and the longer the denoising. This can lead to better results but also can lead to 'blurring' effects if there is
    not enough continuity over these dimensions. For example, if the time interval between time steps is too long and
    consecutive images are too different from each other, it is often better to interpret that dimension as a batch
    dimension. In our experience, it is often better to trade speed for signal: if you can speed up imaging and
    augment signal continuity, even if the signal-to-noise ratio per image is worse, the fact that there are more
    images with different noise patterns but a highly continuous signal, will help achieve better denoising
    performance.
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self._axes = None

        self.tab_layout = QVBoxLayout()
        self.tab_layout.setAlignment(Qt.AlignBottom)

        self.thumbnail_sliders_layout = QHBoxLayout()

        # Explanation text
        self.tab_layout.addWidget(QReadMoreLessLabel(self, self.__doc__))

        # Dimension list, batch and channel dimension selections
        self.dimensions_tree = QTreeWidget()
        self.dimensions_tree.setHeaderLabels(['Dimensions', 'T', 'Z', 'Y', 'X'])
        self.dimensions_tree.itemClicked.connect(self.onTreeClicked)
        self.dimensions_tree.setColumnWidth(0, 150)

        self.tab_layout.addWidget(self.dimensions_tree)

        self.setLayout(self.tab_layout)

    @property
    def axes(self):
        return self._axes

    @property
    def spatiotemporal_axes(self):
        return self._get_current_row_values(1)

    @property
    def batch_axes(self):
        return self._get_current_row_values(2)

    @property
    def channel_axes(self):
        return self._get_current_row_values(3)

    def _get_current_row_values(self, row_index):

        response = []
        root = self.dimensions_tree.invisibleRootItem()

        if root.childCount():
            column_count = root.child(0).columnCount() - 1
            for i in range(1, column_count + 1):
                item = root.child(row_index)
                response.append(bool(item.checkState(i)))

        return None if response == [] else response

    @property
    def dimensions(self):
        # TODO: change this to return a dict about dimensions
        response = []
        root = self.dimensions_tree.invisibleRootItem()

        if root.childCount():
            child_count = root.childCount()
            for i in range(child_count):
                item = root.child(i)
                response.append(item.text(0))

        return None if response == [] else response

    @dimensions.setter
    def dimensions(self, dimensions_metadata):
        if dimensions_metadata is None or dimensions_metadata == FileMetadata():
            self.remove_items_from_tree()
            self.parent.enable_disable_a_tab(self.__class__, False)
            return

        self.parent.enable_disable_a_tab(self.__class__, True)

        self.dimensions_tree.clear()

        for _ in range(len(dimensions_metadata.shape) + 1):
            self.dimensions_tree.header().showSection(_)

        for _ in range(
            len(dimensions_metadata.shape) + 1, self.dimensions_tree.header().count()
        ):
            self.dimensions_tree.header().hideSection(_)

        self.dimensions_tree.setHeaderLabels(
            ['Dimensions'] + [axis for axis in dimensions_metadata.axes]
        )
        self._axes = dimensions_metadata.axes

        self.size_row = QTreeWidgetItem(
            self.dimensions_tree,
            ['size'] + [str(axis_length) for axis_length in dimensions_metadata.shape],
        )

        self.spatiotemporal_row = QTreeWidgetItem(
            self.dimensions_tree,
            ['spatiotemporal'] + ['' for _ in dimensions_metadata.axes],
        )
        self.batch_row = QTreeWidgetItem(
            self.dimensions_tree, ['batch'] + ['' for _ in dimensions_metadata.axes]
        )
        self.channel_row = QTreeWidgetItem(
            self.dimensions_tree, ['channel'] + ['' for _ in dimensions_metadata.axes]
        )

        # Initialize the rows
        for ind, axis in enumerate(dimensions_metadata.batch_axes):
            self.spatiotemporal_row.setCheckState(ind + 1, Qt.Checked)

        for ind, axis in enumerate(dimensions_metadata.batch_axes):
            self.batch_row.setCheckState(ind + 1, Qt.Checked if axis else Qt.Unchecked)
            if axis:
                self.spatiotemporal_row.setCheckState(ind + 1, Qt.Unchecked)

        for ind, axis in enumerate(dimensions_metadata.channel_axes):
            self.channel_row.setCheckState(
                ind + 1, Qt.Checked if axis else Qt.Unchecked
            )
            if axis:
                self.spatiotemporal_row.setCheckState(ind + 1, Qt.Unchecked)

        self.handle_special_cases()

    @Slot(QTreeWidgetItem, int)
    def onTreeClicked(self, it, col):
        nb_spatiotemporal_dimensions = sum(int(x) for x in self.spatiotemporal_axes)
        clicked_row = ["spatiotemporal", "batch", "channel"].index(it.text(0))

        rows = [self.spatiotemporal_row, self.batch_row, self.channel_row]
        rows.pop(clicked_row)
        other_rows = rows

        # if clicked an item to check it
        if bool(it.checkState(col)):
            # clicked on spatiotemporal row
            if it.text(0) == "spatiotemporal":
                if nb_spatiotemporal_dimensions > 4:
                    it.setCheckState(col, Qt.Unchecked)
                else:
                    self.handle_exclusive_states(other_rows=other_rows, col_idx=col)
            # clicked on batch or channel row
            else:
                if (
                    nb_spatiotemporal_dimensions == 1
                    and self.spatiotemporal_row.checkState(col)
                ):
                    it.setCheckState(col, Qt.Unchecked)
                else:
                    self.handle_exclusive_states(other_rows=other_rows, col_idx=col)
        # if clicked an item to uncheck it
        else:
            # if other rows unchecked, check this row back
            for other_row in other_rows:
                if other_row.checkState(col) == Qt.Checked:
                    break
            else:
                it.setCheckState(col, Qt.Checked)

    def handle_special_cases(self):
        nb_spatiotemporal_dimensions = sum(int(x) for x in self.spatiotemporal_axes)

        if (
            nb_spatiotemporal_dimensions == 4
            and "T" in self.axes
            and not (
                self.batch_axes[self.axes.find("T")]
                or self.channel_axes[self.axes.find("T")]
            )
        ):
            self.batch_row.setCheckState(self.axes.find("T") + 1, Qt.Checked)
            self.spatiotemporal_row.setCheckState(self.axes.find("T") + 1, Qt.Unchecked)

    @staticmethod
    def handle_exclusive_states(other_rows, col_idx):
        for other_row in other_rows:
            if other_row.checkState(col_idx) == Qt.Checked:
                other_row.setCheckState(col_idx, Qt.Unchecked)

    def remove_items_from_tree(self):
        root = self.dimensions_tree.invisibleRootItem()
        child_count = root.childCount()
        for i in reversed(range(child_count)):
            item = root.child(i)

            (item.parent() or root).removeChild(item)

    def on_data_model_update(self):
        self.dimensions = (
            self.parent.data_model.images_to_denoise[0][2]
            if len(self.parent.data_model.images_to_denoise) == 1
            else None
        )
