from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)


class AnalysisTab(QWidget):
    """
    Analysis Tab
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        self.snr_estimate_label = QLabel("snr_estimate: ")
        self.tab_layout.addWidget(self.snr_estimate_label)

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)
