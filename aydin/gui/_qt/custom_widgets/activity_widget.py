"""Activity log widget for displaying and managing console output."""

import re
from pathlib import Path

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aydin.util.log.log import Log, aprint

# Pattern matching any ANSI escape sequence (SGR parameters)
_ANSI_SEQ = re.compile(r'\x1b\[[0-9;]*m')
# Pattern to extract 24-bit foreground color: ESC[38;2;R;G;Bm
_ANSI_FG_24BIT = re.compile(r'\x1b\[38;2;(\d+);(\d+);(\d+)m')


def insert_ansi_text(text_edit, text):
    """Insert text with ANSI color codes into a QTextEdit.

    Parses 24-bit ANSI foreground color codes (``ESC[38;2;R;G;Bm``) and
    reset codes (``ESC[0m``) and renders them as colored text using
    QTextCharFormat. Plain text (without ANSI codes) is inserted as-is.

    Parameters
    ----------
    text_edit : QTextEdit
        The text edit widget to insert into.
    text : str
        Text potentially containing ANSI escape sequences.
    """
    cursor = text_edit.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)

    default_fmt = QTextCharFormat()
    current_fmt = QTextCharFormat()

    pos = 0
    for match in _ANSI_SEQ.finditer(text):
        # Insert plain text before this escape sequence
        if match.start() > pos:
            cursor.insertText(text[pos : match.start()], current_fmt)

        seq = match.group()
        color_match = _ANSI_FG_24BIT.fullmatch(seq)
        if color_match:
            r, g, b = (
                int(color_match.group(1)),
                int(color_match.group(2)),
                int(color_match.group(3)),
            )
            current_fmt = QTextCharFormat()
            current_fmt.setForeground(QColor(r, g, b))
        else:
            # Reset or any other sequence -> revert to default
            current_fmt = QTextCharFormat(default_fmt)

        pos = match.end()

    # Insert remaining text after the last escape sequence
    if pos < len(text):
        cursor.insertText(text[pos:], current_fmt)


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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Add information
        self.infoTextBox = QTextEdit(lineWrapMode=QTextEdit.NoWrap)
        # Use monospace font so arbol's box-drawing characters align properly
        monospace_font = QFont()
        monospace_font.setStyleHint(QFont.StyleHint.Monospace)
        monospace_font.setFamilies(["Menlo", "Consolas", "Courier New", "monospace"])
        self.infoTextBox.setFont(monospace_font)

        # We are using an empty press event otherwise cursor jumps to the point clicked
        self.infoTextBox.mousePressEvent = self.qtextedit_mousepressevent
        self.set_auto_scroll()

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
        self.widget_layout.addLayout(self.info_layout)

        self.setLayout(self.widget_layout)

    def copy_logs_to_clipboard(self):
        """Copy the entire activity log text to the system clipboard."""
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(str(self.infoTextBox.toPlainText()))

    def activity_print(self, string2print):
        """Append text to the activity log, rendering ANSI colors.

        Parameters
        ----------
        string2print : str
            Text to append, may contain ANSI color escape sequences.
        """
        insert_ansi_text(self.infoTextBox, string2print)

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
            # images_to_denoise items:
            # [filename, array, metadata, denoise, filepath, output_folder]
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
