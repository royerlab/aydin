import sys

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QSplitter

from pitl.gui.components.log_console import LogConsole
from pitl.gui.tabs.tabs import Tabs


class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.threadpool = QThreadPool()

        self.title = 'Cool Image Translation'
        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 400
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tabs = Tabs(self, self.threadpool)
        #self.log_console = LogConsole(self)
        self.main_widget = QSplitter(Qt.Vertical)
        self.main_widget.addWidget(self.tabs)
        #self.main_widget.addWidget(self.log_console)
        self.main_widget.setSizes([1, 0])
        self.setCentralWidget(self.main_widget)

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu(' &File')
        searchMenu = mainMenu.addMenu(' &Search')
        helpMenu = mainMenu.addMenu(' &Help')

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)


def run():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
