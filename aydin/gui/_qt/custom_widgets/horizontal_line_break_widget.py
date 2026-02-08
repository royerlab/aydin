"""Horizontal line separator widget."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QHBoxLayout, QWidget


class QHorizontalLineBreakWidget(QWidget):
    """A horizontal line separator widget for visual grouping in layouts.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    """

    def __init__(self, parent):
        super(QHorizontalLineBreakWidget, self).__init__(parent)
        self.parent = parent

        self.main_layout = QHBoxLayout()

        self.horizontal_line_break = QFrame(
            frameShape=QFrame.HLine, frameShadow=QFrame.Sunken
        )
        self.main_layout.addWidget(self.horizontal_line_break)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.main_layout)
