from qtpy.QtWidgets import QHBoxLayout, QFrame, QWidget
from qtpy.QtCore import Qt


class QVerticalLineBreakWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.layout = QHBoxLayout()

        self.vertical_line_break = QFrame(
            frameShape=QFrame.VLine, frameShadow=QFrame.Sunken
        )
        self.layout.addWidget(self.vertical_line_break)
        self.layout.setAlignment(Qt.AlignLeft)
        self.setLayout(self.layout)
