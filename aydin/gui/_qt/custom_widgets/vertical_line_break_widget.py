from qtpy.QtWidgets import QHBoxLayout, QFrame, QWidget
from qtpy.QtCore import Qt


class QVerticalLineBreakWidget(QWidget):
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
