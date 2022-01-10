from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QCheckBox,
)

from aydin.util.log.log import Log, lprint


class ActivityWidget(QWidget):
    """Qt dialog window for displaying logs.

    Attributes
    ----------
    info_layout : qtpy.QtWidgets.QHBoxLayout
        Layout widget.
    infoTextBox : qtpy.QtWidgets.QTextEdit
        Text box.
    widget_layout : qtpy.QtWidgets.QVBoxLayout
        Layout widget for the entire the dialog.
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.parent = parent

        self.widget_layout = QVBoxLayout()
        self.setMinimumHeight(120)

        # Add information
        self.infoTextBox = QTextEdit(lineWrapMode=QTextEdit.NoWrap)

        # We are using an empty press event otherwise cursor jumps to the point clicked
        self.infoTextBox.mousePressEvent = self.qtextedit_mousepressevent
        self.set_auto_scroll()

        Log.gui_print = self.activity_print

        # Add text copy and clear buttons
        self.info_copy_button = QPushButton("Copy")
        self.info_copy_button.clicked.connect(self.copy_logs_to_clipboard)
        self.info_clear_button = QPushButton("Clear")
        self.info_clear_button.clicked.connect(self.clear_activity)
        self.info_save_button = QPushButton("Save")
        self.info_save_button.clicked.connect(self.save_activity)
        self.autoscroll_checkbox = QCheckBox("Auto Scroll")
        self.autoscroll_checkbox.setChecked(True)
        self.autoscroll_checkbox.stateChanged.connect(
            self.handle_autoscroll_checkbox_state_changed
        )

        self.info_layout = QHBoxLayout()
        self.info_layout.addWidget(self.infoTextBox, 1)
        self.info_buttons_layout = QVBoxLayout()
        self.info_buttons_layout.setAlignment(Qt.AlignTop)
        self.info_buttons_layout.addWidget(self.info_copy_button, 0, Qt.AlignTop)
        self.info_buttons_layout.addWidget(self.info_clear_button, 0, Qt.AlignTop)
        self.info_buttons_layout.addWidget(self.info_save_button, 0, Qt.AlignTop)
        self.info_buttons_layout.addWidget(self.autoscroll_checkbox, 0, Qt.AlignTop)
        self.info_layout.addLayout(self.info_buttons_layout)
        self.info_layout.setAlignment(Qt.AlignTop)
        self.widget_layout.addLayout(self.info_layout)

        self.setLayout(self.widget_layout)

    def copy_logs_to_clipboard(self):
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(str(self.infoTextBox.toPlainText()))

    def activity_print(self, string2print):
        self.infoTextBox.insertPlainText(string2print)

    def clear_activity(self):
        self.infoTextBox.clear()

    def handle_autoscroll_checkbox_state_changed(self):
        if self.autoscroll_checkbox.isChecked():
            self.set_auto_scroll()
        else:
            self.infoTextBox.verticalScrollBar().rangeChanged.disconnect()

    def set_auto_scroll(self):
        self.infoTextBox.verticalScrollBar().rangeChanged.connect(
            lambda min, max: self.infoTextBox.verticalScrollBar().setSliderPosition(max)
        )

    def save_activity(self):
        log_string = str(self.infoTextBox.toPlainText())
        image_name = None
        for idx, image2denoise in enumerate(
            self.parent.tabs["Image(s)"].images_to_denoise
        ):
            if image2denoise:
                image_name = self.parent.tabs["Image(s)"].images[idx]
                break

        path = None
        for idx, filename in enumerate(self.parent.tabs["File(s)"].filenames):
            if image_name in filename:
                path = self.parent.tabs["File(s)"].filepaths[idx]

        if path is None or path == "":
            lprint("Cannot write the logs into a file")

        logfile_path = f"{path[:path.rfind('.')]}.txt"

        with open(logfile_path, "w+") as logfile:
            logfile.write(log_string)

    def qtextedit_mousepressevent(self, event):
        # We are using an empty press event otherwise cursor jumps to the point clicked
        pass
