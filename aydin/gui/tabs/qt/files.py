from pathlib import Path
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QTreeWidgetItem,
    QTreeWidget,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
)

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.qtreewidget_utils import iter_tree_widget


class FilesTab(QWidget):
    """
    The first step is to load image files into Aydin.

    Drag and drop files or select them with the 'Add Files' button. You can see the files added, and remove files if
    needed. If a single file consists of multiple channels, you can split the image into separate images,
    one per channel, by ticking the corresponding box. If all the images added have the same shape and data type you
    have the option to 'hyperstack' all listed images together along a new dimension. Hyperstacking is selected by
    default if the image shapes and data types are compatible.
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self._hyperstack_last_fail_state_metadatas = None

        self.tab_layout = QVBoxLayout()

        # Explanation text
        self.explanation_text = QLabel(self.__doc__, self)
        self.tab_layout.addWidget(self.explanation_text)

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.label_and_removebutton_layout = QHBoxLayout()
        label = QLabel("Click on column header to check/uncheck all rows below.", self)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet("background-color: purple;")
        self.remove_button.clicked.connect(self.remove_selected_items_from_tree)
        self.remove_all_button = QPushButton("Remove All")
        self.remove_all_button.setStyleSheet("background-color: purple;")
        self.remove_button_layout = QHBoxLayout()
        self.remove_button_layout.setAlignment(Qt.AlignRight)
        self.remove_button_layout.addWidget(self.remove_button)
        self.remove_button_layout.addWidget(self.remove_all_button)

        self.label_and_removebutton_layout.addWidget(label)
        self.label_and_removebutton_layout.addLayout(self.remove_button_layout)
        self.tab_layout.addLayout(self.label_and_removebutton_layout)

        self.file_list_tree_widget = QTreeWidget()
        self.file_list_tree_widget.setHeaderLabels(
            ['File Name', 'Split Channels', 'axes', 'shape', 'dtype', 'Path']
        )
        self.file_list_tree_widget.header().sectionClicked.connect(
            self.onSectionClicked
        )
        self.file_list_tree_widget.setColumnWidth(0, 400)
        self.file_list_tree_widget.setColumnWidth(3, 400)
        self.file_list_tree_widget.header().setSectionsClickable(True)
        self.file_list_tree_widget.dropEvent = self.dropEvent
        self.file_list_tree_widget.dragMoveEvent = self.dragMoveEvent
        self.file_list_tree_widget.dragEnterEvent = self.dragEnterEvent
        self.file_list_tree_widget.itemChanged.connect(self.onTreeItemChanged)
        self.remove_all_button.clicked.connect(self.remove_all_items_from_tree)

        self.tab_layout.addWidget(self.file_list_tree_widget)

        # Checkbox of hyperstacking
        self.hyperstack_checkbox = QCheckBox(
            "Merge files into a single image(Hyperstack)", self
        )
        self.hyperstack_checkbox.toggled.connect(self.on_hyperstack_checkbox_toggled)
        self.tab_layout.addWidget(self.hyperstack_checkbox)

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)

    @Slot(int)
    def onSectionClicked(self, column):
        if column == 0:
            return
        for idx, item in enumerate(
            iter_tree_widget(self.file_list_tree_widget.invisibleRootItem())
        ):
            if idx == 0:
                continue
            elif idx == 1:
                state_to_be_set = (
                    Qt.Unchecked
                    if item.checkState(column) == Qt.Checked
                    else Qt.Checked
                )
                # self.check_for_split_changes()

            item.setCheckState(column, state_to_be_set)

    def onTreeItemChanged(self, item, column):
        if column == 1:
            if (
                self.parent.data_model.set_split_channels(
                    item.text(0), item.text(5), item.checkState(column)
                )
                == -1
            ):
                item.setCheckState(column, False)

    def remove_selected_items_from_tree(self):
        root = self.file_list_tree_widget.invisibleRootItem()
        for item in self.file_list_tree_widget.selectedItems():
            self.parent.data_model.remove_filepaths([item.text(5)])

            (item.parent() or root).removeChild(item)

    def remove_all_items_from_tree(self):
        self.file_list_tree_widget.clear()
        self.parent.data_model.clear_filepaths()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open File(s)", "", "All Files (*)", options=options
        )

        if files:
            self.parent.data_model.add_filepaths(files)
            self.parent.tabwidget.setCurrentIndex(1)

    def on_hyperstack_checkbox_toggled(self):
        response = self.parent.data_model.set_hyperstack(
            self.hyperstack_checkbox.isChecked()
        )
        if response == -1:
            self.hyperstack_checkbox.setChecked(False)

    def on_data_model_update(self):
        filepaths = self.parent.data_model.filepaths

        self.file_list_tree_widget.clear()

        if len(filepaths) == 0:
            return

        for path, (array, metadata) in filepaths.items():
            cg1 = QTreeWidgetItem(
                self.file_list_tree_widget,
                [
                    Path(path).name,
                    ' ',
                    metadata.axes,
                    str(metadata.shape),
                    str(metadata.dtype),
                    path,
                ],
            )
            cg1.setCheckState(1, Qt.Unchecked)
