"""Activity log widget for displaying and managing console output."""

from pathlib import Path

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aydin.util.log.log import Log, aprint


class ActivityWidget(QWidget):
    """Widget for displaying and managing the activity log output.

    Provides a scrollable text area that captures console output via the
    logging system, with buttons for copying, clearing, and saving the log
    content, and an auto-scroll toggle.

    Parameters
    ----------
    parent : MainPage
        The parent MainPage widget.

    Attributes
    ----------
    infoTextBox : QTextEdit
        Text area displaying the log output.
    widget_layout : QVBoxLayout
        Main layout for the widget.
    info_layout : QHBoxLayout
        Layout containing the text box and action buttons.
    """

    def __init__(self, parent):
        """Initialize the activity widget with log text area and control buttons.

        Parameters
        ----------
        parent : MainPage
            The parent MainPage widget.
        """
        super(ActivityWidget, self).__init__(parent)

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
        """Copy the entire activity log text to the system clipboard."""
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(str(self.infoTextBox.toPlainText()))

    def activity_print(self, string2print):
        """Append text to the activity log text box.

        Parameters
        ----------
        string2print : str
            Text to append to the log.
        """
        self.infoTextBox.insertPlainText(string2print)

    def clear_activity(self):
        """Clear all text from the activity log."""
        self.infoTextBox.clear()

    def handle_autoscroll_checkbox_state_changed(self):
        """Enable or disable auto-scrolling based on the checkbox state."""
        if self.autoscroll_checkbox.isChecked():
            self.set_auto_scroll()
        else:
            try:
                self.infoTextBox.verticalScrollBar().rangeChanged.disconnect()
            except (TypeError, RuntimeError):
                # Signal may not be connected or already disconnected
                pass

    def set_auto_scroll(self):
        """Connect the scroll bar to auto-scroll to the bottom on new content."""
        self.infoTextBox.verticalScrollBar().rangeChanged.connect(
            lambda min, max: self.infoTextBox.verticalScrollBar().setSliderPosition(max)
        )

    def save_activity(self):
        """Save the activity log to a text file alongside the first denoised image."""
        log_string = str(self.infoTextBox.toPlainText())

        # Navigate to data_model via parent (MainPage)
        data_model = self.parent.data_model
        images_to_denoise = data_model.images_to_denoise

        path = None
        if images_to_denoise:
            # images_to_denoise items: [filename, array, metadata, denoise, filepath, output_folder]
            path = images_to_denoise[0].filepath

        if not path:
            # Fallback: ask the user to choose a save location
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Activity Log", "aydin_log.txt", "Text Files (*.txt)"
            )
            if not save_path:
                aprint("Cannot write the logs into a file")
                return
            logfile_path = save_path
        else:
            logfile_path = str(Path(path).with_suffix('.txt'))

        with open(logfile_path, "w") as logfile:
            logfile.write(log_string)

    def qtextedit_mousepressevent(self, event):
        """No-op mouse press handler to prevent cursor jumping in the log.

        Parameters
        ----------
        event : QMouseEvent
            The mouse press event (ignored).
        """
        pass
