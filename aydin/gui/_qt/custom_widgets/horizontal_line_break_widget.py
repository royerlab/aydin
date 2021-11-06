from qtpy.QtWidgets import QHBoxLayout, QFrame, QWidget
from qtpy.QtCore import Qt


class QHorizontalLineBreakWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.layout = QHBoxLayout()

        self.horizontal_line_break = QFrame(
            frameShape=QFrame.HLine, frameShadow=QFrame.Sunken
        )
        self.layout.addWidget(self.horizontal_line_break)
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
