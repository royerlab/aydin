import sys
from qtpy.QtCore import QThreadPool
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMainWindow, QAction, QApplication, QStatusBar
import qdarkstyle

from aydin.gui.main_page import MainPage
from aydin.gui.resources.json_resource_loader import absPath
from aydin.util.log.log import lprint


class App(QMainWindow):
    """GUI app"""

    def __init__(self, ver):
        super(QMainWindow, self).__init__()

        self.version = ver

        self.setAydinWindowIcon()

        self.threadpool = QThreadPool(self)

        self.title = "Aydin Studio - image denoising, but chill..."

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        height, width = self.screenRect.height(), self.screenRect.width()

        self.width = width // 3
        self.height = height // 2
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2

        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Status bar
        self.statusBar = QStatusBar(self)
        self.statusBar.showMessage(f"aydin, version: {ver}")
        self.setStatusBar(self.statusBar)

        self.main_widget = MainPage(self, self.threadpool, self.statusBar)
        self.setCentralWidget(self.main_widget)

        # Menu bar
        self.setupMenubar()

    def closeEvent(self, event):
        lprint("closeEvent of mainwindow is called")
        app = QApplication.instance()
        app.quit()

    def setupMenubar(self):
        """Method to populate menubar."""

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu(' &File')
        runMenu = mainMenu.addMenu(' &Run')
        preferencesMenu = mainMenu.addMenu(' &Preferences')
        helpMenu = mainMenu.addMenu(' &Help')

        # File Menu
        startPageButton = QAction('Add File(s)', self)
        startPageButton.setStatusTip('Add new files')
        startPageButton.triggered.connect(
            self.main_widget.tabs["File(s)"].openFileNamesDialog
        )
        fileMenu.addAction(startPageButton)

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Run Menu
        startButton = QAction('Start', self)
        startButton.setStatusTip('Start denoising')
        startButton.triggered.connect(
            self.main_widget.processing_job_runner.prep_and_run
        )
        runMenu.addAction(startButton)

        saveOptionsJSONButton = QAction('Save Options JSON', self)
        saveOptionsJSONButton.setStatusTip('Save options JSON')
        saveOptionsJSONButton.triggered.connect(
            lambda: self.main_widget.save_options_json()
        )
        runMenu.addAction(saveOptionsJSONButton)

        # Preferences Menu
        self.basicModeButton = QAction('Basic mode', self)
        self.basicModeButton.setEnabled(False)
        self.basicModeButton.setStatusTip('Switch to basic mode')
        self.basicModeButton.triggered.connect(
            lambda: self.main_widget.toggle_basic_advanced_mode()
        )
        preferencesMenu.addAction(self.basicModeButton)

        self.advancedModeButton = QAction('Advanced mode', self)
        self.advancedModeButton.setStatusTip('Switch to advanced mode')
        self.advancedModeButton.triggered.connect(
            lambda: self.main_widget.toggle_basic_advanced_mode()
        )
        preferencesMenu.addAction(self.advancedModeButton)

        # Help Menu
        versionButton = QAction("ver" + self.version, self)
        helpMenu.addAction(versionButton)

    def setAydinWindowIcon(self):
        self.setWindowIcon(QIcon(absPath("aydin_icon.png")))


def run(ver):
    """Method to run GUI

    Parameters
    ----------
    ver : str
        string of aydin version number

    """
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App(ver)
    ex.show()
    sys.exit(app.exec())
