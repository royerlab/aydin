"""Shared fixtures for GUI tests.

Qt-dependent fixtures (mock_main_page, etc.) are only available when a
display server or Xvfb is running.  Non-Qt tests under aydin/gui/ (e.g.
json_resource_loader, data_model) continue to work without a display.
"""

import sys
from unittest.mock import Mock

import numpy
import pytest

from aydin.io.io import FileMetadata

# Guard Qt imports so that non-Qt tests under aydin/gui/ are not skipped
# when no display is available.
try:
    from qtpy.QtWidgets import QWidget

    _QT_AVAILABLE = True
except (ImportError, RuntimeError):
    _QT_AVAILABLE = False


if _QT_AVAILABLE:
    from aydin.gui.tabs.data_model import DataModel

    class MockMainPage(QWidget):
        """Minimal stand-in for MainPage that provides the interface tabs expect.

        Extends QWidget so that tabs can call ``super().__init__(parent)``
        with a real QWidget parent.
        """

        def __init__(self):
            super().__init__()
            self.data_model = DataModel(self)
            self.threadpool = Mock()
            self.tabwidget = Mock()
            self.tabs = {}

        # No-op callbacks that DataModel and tabs invoke on the parent
        def enable_disable_a_tab(self, tab_class, enabled):
            pass

        def filestab_changed(self):
            pass

        def imagestab_changed(self):
            pass

        def dimensionstab_changed(self):
            pass

        def croppingtabs_changed(self):
            pass


def make_file_metadata(axes, shape, dtype=numpy.float32):
    """Create a FileMetadata with the given axes, shape, and dtype.

    Parameters
    ----------
    axes : str
        Axis labels (e.g. 'YX', 'ZYX', 'TZYX').
    shape : tuple of int
        Shape of the image array.
    dtype : numpy.dtype, optional
        Data type. Default is float32.

    Returns
    -------
    FileMetadata
        Populated metadata instance.
    """
    meta = FileMetadata()
    meta.axes = axes
    meta.shape = shape
    meta.dtype = numpy.dtype(dtype)
    meta.batch_axes = tuple(False for _ in axes)
    meta.channel_axes = tuple(False for _ in axes)
    meta.format = 'synthetic'
    return meta


@pytest.fixture
def mock_main_page(qtbot):
    """Create a MockMainPage and register it with qtbot."""
    if not _QT_AVAILABLE:
        pytest.skip("Qt not available")
    page = MockMainPage()
    qtbot.addWidget(page)
    return page


@pytest.fixture
def sample_metadata():
    """Return a 2D YX FileMetadata without loading a real file."""
    return make_file_metadata('YX', (512, 512))


@pytest.fixture
def sample_metadata_3d():
    """Return a 3D ZYX FileMetadata."""
    return make_file_metadata('ZYX', (32, 256, 256))


@pytest.fixture
def sample_metadata_4d():
    """Return a 4D TZYX FileMetadata."""
    return make_file_metadata('TZYX', (10, 32, 256, 256))


@pytest.fixture(autouse=True)
def _guard_stdout_stderr():
    """Assert that sys.stdout and sys.stderr are not leaked between tests."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    yield
    sys.stdout = original_stdout
    sys.stderr = original_stderr
