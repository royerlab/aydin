"""Aydin Studio main application window and entry point."""

import platform
import sys


def _check_linux_qt_dependencies():
    """Check for required system libraries on Linux before Qt initialization.

    Qt 6.5+ requires libxcb-cursor0 on Linux. If missing, Qt will abort with
    an unhelpful error. This function detects the issue early and provides
    clear installation instructions.
    """
    if platform.system() != 'Linux':
        return

    import ctypes
    import ctypes.util

    # Try to load libxcb-cursor
    lib_name = ctypes.util.find_library('xcb-cursor')
    if lib_name:
        try:
            ctypes.CDLL(lib_name)
            return  # Library found and loadable
        except OSError:
            pass

    # Try common library paths directly
    for lib_path in ['libxcb-cursor.so.0', 'libxcb-cursor.so']:
        try:
            ctypes.CDLL(lib_path)
            return  # Library found and loadable
        except OSError:
            continue

    # Library not found - print helpful message and exit
    print(
        """
================================================================================
ERROR: Missing required system library for Qt GUI

Aydin's GUI requires 'libxcb-cursor0' on Linux (required since Qt 6.5).

To fix this, install the library using your package manager:

  Ubuntu/Debian:
    sudo apt install libxcb-cursor0

  Fedora/RHEL:
    sudo dnf install xcb-util-cursor

  Arch Linux:
    sudo pacman -S xcb-util-cursor

After installing, run 'aydin' again.

For more information, see the Aydin installation documentation:
https://royerlab.github.io/aydin/
================================================================================
"""
    )
    sys.exit(1)


# Check dependencies before importing Qt
_check_linux_qt_dependencies()

import qdarkstyle
from qtpy.QtCore import QThreadPool
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QAction, QApplication, QMainWindow, QStatusBar

from aydin.gui.main_page import MainPage
from aydin.gui.resources.json_resource_loader import absPath
from aydin.util.log.log import aprint


class App(QMainWindow):
    """Main application window for Aydin Studio.

    Sets up the main window geometry, status bar, menu bar, and central
    widget (MainPage). Acts as the top-level container for the GUI.

    Parameters
    ----------
    ver : str
        Aydin version string displayed in the status bar and help menu.
    """

    def __init__(self, ver):
        super(App, self).__init__()

        self.version = ver

        self.setAydinWindowIcon()

        self.threadpool = QThreadPool(self)

        self.title = "Aydin Studio - image denoising, but chill..."

        screen = QApplication.primaryScreen()
        self.screenRect = screen.availableGeometry()
        height, width = self.screenRect.height(), self.screenRect.width()

        self.width = width // 3
        self.height = height // 2
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Status bar
        self.statusBar = QStatusBar(self)
        self.statusBar.showMessage(f"aydin, version: {ver}")
        self.setStatusBar(self.statusBar)

        self.main_widget = MainPage(self, self.threadpool, self.statusBar)
        self.setCentralWidget(self.main_widget)

        # Menu bar
        self.setupMenubar()

    def closeEvent(self, event):
        """Handle window close by quitting the application.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        aprint("closeEvent of mainwindow is called")
        app = QApplication.instance()
        app.quit()

    def setupMenubar(self):
        """Populate the menu bar with File, Run, Preferences, and Help menus."""

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

        loadPretrainedModelButton = QAction('Load Pretrained Model', self)
        loadPretrainedModelButton.setStatusTip('Load Pretrained Model')
        loadPretrainedModelButton.triggered.connect(
            lambda: self.main_widget.load_pretrained_model()
        )
        runMenu.addAction(loadPretrainedModelButton)

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
        """Set the Aydin icon for the application window."""
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
