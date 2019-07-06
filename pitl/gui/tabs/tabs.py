from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QTabWidget

from pitl.gui.tabs.AboutTab import AboutTab
from pitl.gui.tabs.Noise2SelfTab import Noise2SelfTab
from pitl.gui.tabs.Noise2TruthTab import Noise2TruthTab


class Tabs(QTabWidget):

    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        # Initialize tab screen
        self.tab3 = QWidget()

        # Add tabs
        self.addTab(Noise2SelfTab(self, f), "Noise2Self")
        self.addTab(Noise2TruthTab(self), "Noise2Truth")
        self.addTab(AboutTab(self), "About")

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
