"""Aydin Studio main application window and entry point."""

import os
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
    print("""
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
""")
    sys.exit(1)


def _maybe_relaunch_as_macos_app():
    """On macOS, relaunch through a .app bundle for proper dock/menu identity.

    Without a .app bundle, macOS shows 'python3.x' in the dock tooltip and
    'python' in the menu bar because the window server identifies the app by
    its executable path.  This function creates a minimal .app bundle in
    ``~/Library/Application Support/Aydin/`` and relaunches through the macOS
    ``open`` command, which registers the app properly with the window server.

    The .app contains an ``Info.plist`` (with CFBundleName, icon, etc.) and a
    tiny launcher shell script that ``exec``'s back into the same ``aydin``
    entry point with ``AYDIN_APP_BUNDLE=1`` to prevent infinite re-launch.

    On non-macOS platforms this is a no-op.  On macOS, if the relaunch succeeds
    this function **does not return** (it waits for the child via ``open -W``
    and calls ``sys.exit``).  If anything fails it returns silently and the app
    launches the normal way.
    """
    if sys.platform != 'darwin':
        return

    # Already running inside the .app wrapper — continue normally.
    if os.environ.get('AYDIN_APP_BUNDLE'):
        return

    # Running as a PyInstaller bundle — it already has proper dock identity.
    if getattr(sys, 'frozen', False):
        return

    try:
        import shutil
        import subprocess

        from aydin.gui.resources.json_resource_loader import abs_path

        # Stable location so we don't recreate every launch
        app_support = os.path.join(
            os.path.expanduser('~'),
            'Library',
            'Application Support',
            'Aydin',
        )
        app_dir = os.path.join(app_support, 'Aydin Studio.app')
        contents = os.path.join(app_dir, 'Contents')
        macos = os.path.join(contents, 'MacOS')
        resources = os.path.join(contents, 'Resources')

        os.makedirs(macos, exist_ok=True)
        os.makedirs(resources, exist_ok=True)

        # Copy icon (PNG — modern macOS accepts PNG via CFBundleIconFile)
        icon_src = abs_path('aydin_icon.png')
        if os.path.exists(icon_src):
            shutil.copy2(icon_src, os.path.join(resources, 'aydin_icon.png'))

        # Write Info.plist
        with open(os.path.join(contents, 'Info.plist'), 'w') as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
                ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0">\n'
                '<dict>\n'
                '  <key>CFBundleExecutable</key>\n'
                '  <string>launcher</string>\n'
                '  <key>CFBundleName</key>\n'
                '  <string>Aydin Studio</string>\n'
                '  <key>CFBundleDisplayName</key>\n'
                '  <string>Aydin Studio</string>\n'
                '  <key>CFBundleIdentifier</key>\n'
                '  <string>org.royerlab.aydin</string>\n'
                '  <key>CFBundleIconFile</key>\n'
                '  <string>aydin_icon</string>\n'
                '  <key>CFBundlePackageType</key>\n'
                '  <string>APPL</string>\n'
                '  <key>CFBundleInfoDictionaryVersion</key>\n'
                '  <string>6.0</string>\n'
                '  <key>NSHighResolutionCapable</key>\n'
                '  <true/>\n'
                '</dict>\n'
                '</plist>\n'
            )

        # Write launcher script — execs back into the same aydin entry point.
        # The AYDIN_APP_BUNDLE env var prevents infinite re-launch.
        entry_point = os.path.abspath(sys.argv[0])
        launcher_path = os.path.join(macos, 'launcher')
        with open(launcher_path, 'w') as f:
            f.write(
                '#!/bin/bash\n'
                'export AYDIN_APP_BUNDLE=1\n'
                f'exec "{sys.executable}" "{entry_point}" "$@"\n'
            )
        os.chmod(launcher_path, 0o755)

        # Launch the .app and wait for it to quit, then exit with same code.
        # If `open` fails (non-zero return), fall through to normal launch.
        ret = subprocess.call(['open', '-W', '-a', app_dir])
        if ret == 0:
            sys.exit(0)

    except Exception:
        pass  # Fall through to normal (non-bundled) launch


# --- Pre-Qt platform setup (must run before any Qt/Cocoa imports) ----------
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

    def __init__(self, ver, napari_viewer=None):
        """Initialize the main application window with menu bar and central widget.

        Parameters
        ----------
        ver : str
            Aydin version string displayed in the status bar and help menu.
        napari_viewer : napari.viewer.Viewer, optional
            When launched from napari, the existing viewer instance.
            Prevents ``app.quit()`` on close and adjusts napari integration.
        """
        super(App, self).__init__()

        self.version = ver
        self.napari_viewer = napari_viewer

        self.set_aydin_window_icon()

        self.threadpool = QThreadPool(self)

        self.title = "Aydin Studio — Self-supervised image denoising"

        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        screen_height, screen_width = screen_rect.height(), screen_rect.width()

        win_width = min(screen_width * 2 // 3, 1200)
        win_height = min(screen_height * 2 // 3, 800)
        left = (screen_width - win_width) // 2
        top = (screen_height - win_height) // 2

        self.setWindowTitle(self.title)
        self.setMinimumSize(640, 480)
        self.setGeometry(left, top, win_width, win_height)

        # Restore saved geometry if available, but validate against screen
        settings = QSettings("Aydin", "AydinStudio")
        saved_geometry = settings.value("geometry")
        if saved_geometry is not None:
            self.restoreGeometry(saved_geometry)
            # Reset if saved geometry doesn't fit the current screen
            if (
                self.width() > screen_rect.width()
                or self.height() > screen_rect.height()
            ):
                self.setGeometry(left, top, win_width, win_height)

        # Status bar
        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage(f"aydin, version: {ver}")
        self.setStatusBar(self.status_bar)

        self.main_widget = MainPage(
            self, self.threadpool, self.status_bar, napari_viewer=napari_viewer
        )
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
        if self.napari_viewer is None:
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
        self._add_files_action = QAction('Add File(s)', self)
        self._add_files_action.setStatusTip('Add new files')
        self._add_files_action.triggered.connect(
            self.main_widget.tabs["File(s)"].open_file_names_dialog
        )
        fileMenu.addAction(self._add_files_action)

        if self.napari_viewer is not None:
            self._add_layers_action = QAction('Add Layer(s)', self)
            self._add_layers_action.setStatusTip('Import image layers from napari')
            self._add_layers_action.triggered.connect(
                self.main_widget._add_napari_layers
            )
            fileMenu.addAction(self._add_layers_action)

        self._exit_action = QAction(QIcon('exit24.png'), 'Exit', self)
        self._exit_action.setShortcut('Ctrl+Q')
        self._exit_action.setStatusTip('Exit application')
        self._exit_action.triggered.connect(self.close)
        fileMenu.addAction(self._exit_action)

        # Run Menu
        self._start_action = QAction('Start', self)
        self._start_action.setStatusTip('Start denoising')
        self._start_action.triggered.connect(
            self.main_widget.processing_job_runner.prep_and_run
        )
        runMenu.addAction(self._start_action)

        self._save_options_action = QAction('Save Options JSON', self)
        self._save_options_action.setStatusTip('Save options JSON')
        self._save_options_action.triggered.connect(
            lambda: self.main_widget.save_options_json()
        )
        runMenu.addAction(self._save_options_action)

        self._load_model_action = QAction('Load Pretrained Model', self)
        self._load_model_action.setStatusTip('Load Pretrained Model')
        self._load_model_action.triggered.connect(
            lambda: self.main_widget.load_pretrained_model()
        )
        runMenu.addAction(self._load_model_action)

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
        self._version_action = QAction("ver" + self.version, self)
        helpMenu.addAction(self._version_action)

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
    # On macOS, relaunch inside a .app bundle so the dock and menu bar
    # show "Aydin Studio" instead of "python3.x".  If it succeeds this
    # call does not return.  On other platforms it is a no-op.
    _maybe_relaunch_as_macos_app()

    app = QApplication(sys.argv)
    app.setApplicationName('Aydin Studio')
    app.setApplicationDisplayName('Aydin Studio')
    app.setWindowIcon(QIcon(abs_path("aydin_icon.png")))
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    ex = App(ver)
    ex.show()
    sys.exit(app.exec())


def run_from_napari(ver, napari_viewer):
    """Launch Aydin Studio as a window within an existing napari session.

    Does NOT create a ``QApplication`` (napari owns it) and does NOT call
    ``_maybe_relaunch_as_macos_app()``.  Applies the dark stylesheet to the
    window only so it doesn't override napari's own theme.

    Parameters
    ----------
    ver : str
        Aydin version string.
    napari_viewer : napari.viewer.Viewer
        The napari viewer instance for bidirectional integration.

    Returns
    -------
    App
        The Aydin Studio ``App`` window.  The caller must keep a reference
        to prevent garbage collection.
    """
    ex = App(ver, napari_viewer=napari_viewer)
    ex.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    ex.show()
    return ex
