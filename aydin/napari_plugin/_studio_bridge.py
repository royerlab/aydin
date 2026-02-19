"""Bridge between the napari dock widget and Aydin Studio (Tier 2)."""

import napari
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from aydin.napari_plugin._axes_utils import detect_axes_from_napari_layer

# Keep a reference to prevent garbage collection of the Studio window.
_studio_window = None


def launch_studio_from_napari(viewer):
    """Open Aydin Studio pre-populated with selected napari layers.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    Returns
    -------
    App
        The Aydin Studio window.
    """
    global _studio_window

    from aydin import __version__
    from aydin.gui.gui import run_from_napari

    window = run_from_napari(__version__, napari_viewer=viewer)

    # Pre-populate the DataModel with selected (or all) Image layers
    selected_layers = [
        layer
        for layer in viewer.layers.selection
        if isinstance(layer, napari.layers.Image)
    ]
    if not selected_layers:
        selected_layers = [
            layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
        ]

    if selected_layers:
        arrays_dict = {}
        for layer in selected_layers:
            metadata = detect_axes_from_napari_layer(layer, viewer)
            arrays_dict[layer.name] = (layer.data.copy(), metadata)

        window.main_widget.data_model.add_arrays(arrays_dict)

        # Jump to the Dimensions tab so the user starts further in the workflow
        tabwidget = window.main_widget.tabwidget
        dims_tab = window.main_widget.tabs.get("Dimensions")
        if dims_tab is not None:
            tabwidget.setCurrentIndex(tabwidget.indexOf(dims_tab))

    _studio_window = window
    return window


class AydinStudioWidget(QWidget):
    """Napari dock widget that launches Aydin Studio as a separate window.

    Provides a minimal dock panel with a button to open or bring Studio
    to the front.  Studio is launched on first creation and can be
    re-opened if closed.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance (injected by napari).
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        self._label = QLabel('Aydin Studio runs as a separate window.')
        layout.addWidget(self._label)

        self._open_btn = QPushButton('Open Aydin Studio')
        self._open_btn.clicked.connect(self._open_or_raise)
        layout.addWidget(self._open_btn)

        layout.addStretch()
        self.setLayout(layout)

        # Launch Studio immediately
        self._open_or_raise()

    def _open_or_raise(self):
        """Open Studio or bring an existing window to front."""
        global _studio_window
        if _studio_window is not None and _studio_window.isVisible():
            _studio_window.raise_()
            _studio_window.activateWindow()
        else:
            _studio_window = launch_studio_from_napari(self._viewer)
