"""Images tab for inspecting loaded images and selecting which to denoise."""

import numpy
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.qtreewidget_utils import iter_tree_widget
from aydin.util.misc.units import human_readable_byte_size


class ImagesTab(QWidget):
    """
    Next, we inspect the images contained in the added files.

    You can see basic image information such as name, axis label, array shapes, data types, volume in voxels,
    and size in bytes. When several images are loaded, each images is denoised independently which means that
    training and denoising are done per image. For bulk denoising of series of images please use the command line
    interface (CLI).
    """

    def __init__(self, parent):
        super(ImagesTab, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        # Explanation text
        self.explanation_text = QLabel(self.__doc__, self)
        self.tab_layout.addWidget(self.explanation_text)

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        label = QLabel("Click on column header to check/uncheck all rows below.", self)
        self.tab_layout.addWidget(label)

        self.image_list_tree_widget = QTreeWidget()
        self.image_list_tree_widget.setHeaderLabels(
            ['file name', 'denoise', 'axes', 'shape', 'dtype', 'size', 'output folder']
        )

        self.image_list_tree_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.image_list_tree_widget.itemDoubleClicked.connect(self.onItemDoubleClick)

        self.image_list_tree_widget.header().sectionClicked.connect(
            self.onSectionClicked
        )
        self.image_list_tree_widget.header().setSectionsClickable(True)

        self.image_list_tree_widget.setColumnWidth(0, 400)
        self.image_list_tree_widget.setColumnWidth(3, 300)
        self.image_list_tree_widget.setColumnWidth(5, 200)
        self.image_list_tree_widget.itemChanged.connect(self.onTreeItemChanged)
        self.tab_layout.addWidget(self.image_list_tree_widget)

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)

    @property
    def images2denoise(self):
        """Boolean list of which images are marked for denoising.

        Returns
        -------
        list of bool
            One boolean per image in the tree, True if denoise is checked.
        """
        response = []
        root = self.image_list_tree_widget.invisibleRootItem()
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            response.append(item.checkState(1) == Qt.Checked)

        return response

    @Slot(int)
    def onSectionClicked(self, column):
        """Handle column header clicks to bulk-toggle checkboxes or copy output paths.

        Parameters
        ----------
        column : int
            The column index of the clicked header section.
        """
        if column == 0:
            return
        elif column == 1:
            for idx, item in enumerate(
                iter_tree_widget(self.image_list_tree_widget.invisibleRootItem())
            ):
                if idx == 0:
                    continue
                elif idx == 1:
                    state_to_be_set = (
                        Qt.Unchecked
                        if item.checkState(column) == Qt.Checked
                        else Qt.Checked
                    )

                item.setCheckState(column, state_to_be_set)
        elif column == 6:
            for idx, item in enumerate(
                iter_tree_widget(self.image_list_tree_widget.invisibleRootItem())
            ):
                if idx == 0:
                    continue
                item.setText(6, self.image_list_tree_widget.selectedItems()[0].text(6))

    @Slot(QTreeWidgetItem, int)
    def onItemDoubleClick(self, item, column):
        """Allow editing of the output folder column on double-click.

        Parameters
        ----------
        item : QTreeWidgetItem
            The double-clicked tree item.
        column : int
            The column index that was double-clicked.
        """
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        if column == 6:
            self.image_list_tree_widget.editItem(item, column)

    def onTreeItemChanged(self, item, column):
        """Propagate tree item changes to the data model.

        Updates the denoise flag (column 1) or output folder (column 6)
        in the data model when the user modifies them in the tree.

        Parameters
        ----------
        item : QTreeWidgetItem
            The changed tree item.
        column : int
            The column index that changed.
        """
        if column == 1:
            self.parent.data_model.set_image_to_denoise(
                item.text(0), item.checkState(1) == Qt.Checked
            )
        elif column == 6:
            self.parent.data_model.update_image_output_folder(
                item.text(0), item.text(6)
            )

    def remove_items_from_tree(self):
        """Clear all items from the image list tree widget."""
        self.image_list_tree_widget.clear()

    def setFollowingThreeTabsEnabled(self, bool):
        """Enable or disable the Dimensions, Training Crop, and Denoising Crop tabs.

        Parameters
        ----------
        bool : bool
            Whether to enable (True) or disable (False) the tabs.
        """
        for _ in range(3, 6):
            self.parent.tabwidget.setTabEnabled(_, bool)

    def on_data_model_update(self):
        """Rebuild the image list tree from the current data model."""
        imagelist = self.parent.data_model.images
        if imagelist is None or imagelist == []:
            self.remove_items_from_tree()
            return

        self.image_list_tree_widget.clear()

        for filename, array, metadata, denoise, path, output_folder in imagelist:
            qtree_widget_item = QTreeWidgetItem(
                self.image_list_tree_widget,
                [
                    filename,
                    ' ',
                    str(metadata.axes),
                    str(metadata.shape),
                    str(metadata.dtype),
                    human_readable_byte_size(
                        metadata.dtype.itemsize * numpy.prod(metadata.shape)
                    ),
                    output_folder,
                ],
            )

            qtree_widget_item.setCheckState(1, Qt.Checked if denoise else Qt.Unchecked)
            qtree_widget_item.setToolTip(0, path)
