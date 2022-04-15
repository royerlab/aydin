from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout


class AnalysisTab(QWidget):
    """
    Analysis Tab
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)
