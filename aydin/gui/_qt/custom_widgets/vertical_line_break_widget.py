"""Vertical line separator widget."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QHBoxLayout, QWidget


class QVerticalLineBreakWidget(QWidget):
    """A vertical line separator widget for visual grouping in layouts.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    """

    def __init__(self, parent):
        super(QVerticalLineBreakWidget, self).__init__(parent)
        self.parent = parent

        self.main_layout = QHBoxLayout()

        self.vertical_line_break = QFrame(
            frameShape=QFrame.VLine, frameShadow=QFrame.Sunken
        )
        self.main_layout.addWidget(self.vertical_line_break)
        self.main_layout.setAlignment(Qt.AlignLeft)
        self.setLayout(self.main_layout)
