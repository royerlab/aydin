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
from qtpy.QtCore import QSettings, QThreadPool
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QAction, QApplication, QMainWindow, QStatusBar

from aydin.gui.main_page import MainPage
from aydin.gui.resources.json_resource_loader import abs_path
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
        """Initialize the main application window with menu bar and central widget.

        Parameters
        ----------
        ver : str
            Aydin version string displayed in the status bar and help menu.
        """
        super(App, self).__init__()

        self.version = ver

        self.set_aydin_window_icon()

        self.threadpool = QThreadPool(self)

        self.title = "Aydin Studio - image denoising, but chill..."

        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        screen_height, screen_width = screen_rect.height(), screen_rect.width()

        win_width = screen_width // 3
        win_height = screen_height // 2
        left = (screen_width - win_width) // 2
        top = (screen_height - win_height) // 2

        self.setWindowTitle(self.title)
        self.setGeometry(left, top, win_width, win_height)

        # Restore saved geometry if available
        settings = QSettings("Aydin", "AydinStudio")
        saved_geometry = settings.value("geometry")
        if saved_geometry is not None:
            self.restoreGeometry(saved_geometry)

        # Status bar
        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage(f"aydin, version: {ver}")
        self.setStatusBar(self.status_bar)

        self.main_widget = MainPage(self, self.threadpool, self.status_bar)
        self.setCentralWidget(self.main_widget)

        # Menu bar
        self.setup_menubar()

    def closeEvent(self, event):
        """Handle window close by saving geometry, waiting for workers, and quitting.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        aprint("closeEvent of mainwindow is called")

        # Save window geometry for next launch
        settings = QSettings("Aydin", "AydinStudio")
        settings.setValue("geometry", self.saveGeometry())

        self.threadpool.waitForDone(5000)
        app = QApplication.instance()
        app.quit()
        event.accept()

    def setup_menubar(self):
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
            self.main_widget.tabs["File(s)"].open_file_names_dialog
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

    def set_aydin_window_icon(self):
        """Set the Aydin icon for the application window."""
        self.setWindowIcon(QIcon(abs_path("aydin_icon.png")))


def run(ver):
    """Launch the Aydin Studio GUI application.

    Creates a QApplication, applies the dark stylesheet, instantiates the
    main App window, and starts the Qt event loop.

    Parameters
    ----------
    ver : str
        Aydin version string displayed in the window title and status bar.
    """
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    ex = App(ver)
    ex.show()
    sys.exit(app.exec())
