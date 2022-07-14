from qtpy.QtWidgets import QHBoxLayout, QFrame, QWidget
from qtpy.QtCore import Qt


class QHorizontalLineBreakWidget(QWidget):
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
