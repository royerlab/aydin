import numpy
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTreeWidgetItem, QTreeWidget

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
        super(QWidget, self).__init__(parent)
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
            ['file name', 'denoise', 'axes', 'shape', 'dtype', 'size']
        )

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
        response = []
        root = self.image_list_tree_widget.invisibleRootItem()
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            response.append(bool(item.checkState(1)))

        return response

    @Slot(int)
    def onSectionClicked(self, column):
        if column == 0:
            return
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

    def onTreeItemChanged(self, item, column):
        if column == 1:
            self.parent.data_model.set_image_to_denoise(
                item.text(0), item.checkState(1)
            )

    def remove_items_from_tree(self):
        self.image_list_tree_widget.clear()

    def setFollowingThreeTabsEnabled(self, bool):
        for _ in range(3, 6):
            self.parent.tabwidget.setTabEnabled(_, bool)

    def on_data_model_update(self):
        imagelist = self.parent.data_model.images
        if imagelist is None or imagelist == []:
            self.remove_items_from_tree()
            return

        self.image_list_tree_widget.clear()

        for filename, array, metadata, train_on, denoise, path in imagelist:
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
                ],
            )
            qtree_widget_item.setCheckState(1, Qt.Checked if denoise else Qt.Unchecked)
            qtree_widget_item.setToolTip(0, path)
