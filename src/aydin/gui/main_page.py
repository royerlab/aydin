"""Central widget for Aydin Studio containing the tabbed workflow interface."""

import sys
import traceback

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aydin.gui._qt.custom_widgets.activity_widget import ActivityWidget
from aydin.gui._qt.custom_widgets.overlay import Overlay
from aydin.gui._qt.custom_widgets.program_flow_diagram import QProgramFlowDiagramWidget
from aydin.gui._qt.job_runners.denoise_job_runner import DenoiseJobRunner
from aydin.gui.resources.json_resource_loader import JSONResourceLoader
from aydin.gui.tabs.data_model import DataModel
from aydin.gui.tabs.qt.denoise import DenoiseTab
from aydin.gui.tabs.qt.denoising_cropping import DenoisingCroppingTab
from aydin.gui.tabs.qt.dimensions import DimensionsTab
from aydin.gui.tabs.qt.files import FilesTab
from aydin.gui.tabs.qt.images import ImagesTab
from aydin.gui.tabs.qt.processing import ProcessingTab
from aydin.gui.tabs.qt.summary import SummaryTab
from aydin.gui.tabs.qt.training_cropping import TrainingCroppingTab
from aydin.io.utils import get_options_json_path
from aydin.util.log.log import aprint
from aydin.util.misc.json import save_any_json


class MainPage(QWidget):
    """Central widget for Aydin Studio containing the tab-based workflow.

    Manages the navigation bar, tab widget (File(s), Image(s), Dimensions,
    Training Crop, Denoising Crop, Pre/Post-Processing, Denoise), activity
    dock, and the denoising job runner. Supports drag-and-drop file loading.

    Parameters
    ----------
    parent : QMainWindow
        The parent App window.
    threadpool : QThreadPool
        Thread pool used for running denoising jobs in background threads.
    status_bar : QStatusBar
        The application status bar for displaying messages.
    """

    def __init__(self, parent, threadpool, status_bar, napari_viewer=None):
        """Initialize the main page with tabs, navigation, and job runners.

        Parameters
        ----------
        parent : QMainWindow
            The parent App window.
        threadpool : QThreadPool
            Thread pool used for running denoising jobs in background threads.
        status_bar : QStatusBar
            The application status bar for displaying messages.
        napari_viewer : napari.viewer.Viewer, optional
            When launched from napari, the existing viewer instance.
        """
        super(MainPage, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool
        self.status_bar = status_bar
        self.napari_viewer = napari_viewer

        self.disable_spatial_features = False

        self.setAcceptDrops(True)

        self.tooltips = JSONResourceLoader(resource_file_name="tooltips.json")

        self.tabs = {
            "Summary": SummaryTab(self),
            "File(s)": FilesTab(self),
            "Image(s)": ImagesTab(self),
            "Dimensions": DimensionsTab(self),
            "Training Crop": TrainingCroppingTab(self),
            "Denoising Crop": DenoisingCroppingTab(self),
            "Pre/Post-Processing": ProcessingTab(self),
            "Denoise": DenoiseTab(self),
        }

        self.data_model = DataModel(self)

        self.activity_dock = QDockWidget("Activity", self)
        self.activity_dock.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # MainPage layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)

        self.add_activity_dockable()

        # Navbar: flow diagram on the left, action buttons on the right
        self.navbar_layout = QHBoxLayout()
        self.navbar_layout.setContentsMargins(0, 0, 9, 0)
        self.navbar_layout.setSpacing(5)

        # Flow diagram (auto-hides when window is narrow)
        self.flow_diagram_widget = QProgramFlowDiagramWidget(self)
        # Use Ignored horizontal policy so the flow diagram does not
        # force the window to be wider than necessary.  The diagram
        # auto-hides via resizeEvent when there isn't enough room.
        sp = self.flow_diagram_widget.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Ignored)
        self.flow_diagram_widget.setSizePolicy(sp)
        self.navbar_layout.addWidget(self.flow_diagram_widget, 1)

        self.flow_diagram_widget.add_files_button.clicked.connect(
            self.tabs["File(s)"].open_file_names_dialog
        )
        self.flow_diagram_widget.add_files_button.setIcon(
            QApplication.style().standardIcon(QStyle.SP_DirIcon)
        )
        self.flow_diagram_widget.add_files_button.setToolTip(
            self.tooltips.json["main_page"]["open_file_button"]
        )
        self.flow_diagram_widget.load_sample_image_button.setToolTip(
            self.tooltips.json["main_page"]["load_example_button"]
        )

        # In napari mode, replace "Examples" with "Add Layer(s)"
        if self.napari_viewer is not None:
            self.flow_diagram_widget.set_napari_mode(True)
            self.flow_diagram_widget.add_layers_button.clicked.connect(
                self._add_napari_layers
            )
            self.flow_diagram_widget.add_layers_button.setToolTip(
                'Import selected napari image layers (or all if none selected)'
            )

        self.flow_diagram_widget.files_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["File(s)"])
            )
        )
        self.flow_diagram_widget.images_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Image(s)"])
            )
        )
        self.flow_diagram_widget.dimensions_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Dimensions"])
            )
        )
        self.flow_diagram_widget.training_crop_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Training Crop"])
            )
        )
        self.flow_diagram_widget.inference_crop_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Denoising Crop"])
            )
        )
        self.flow_diagram_widget.preprocess_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Pre/Post-Processing"])
            )
        )
        self.flow_diagram_widget.postprocess_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Pre/Post-Processing"])
            )
        )
        self.flow_diagram_widget.denoise_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Denoise"])
            )
        )

        # Tooltips for flow diagram stage buttons
        tips = self.tooltips.json["main_page"]
        self.flow_diagram_widget.files_button.setToolTip(tips["files_button"])
        self.flow_diagram_widget.images_button.setToolTip(tips["images_button"])
        self.flow_diagram_widget.dimensions_button.setToolTip(tips["dimensions_button"])
        self.flow_diagram_widget.training_crop_button.setToolTip(
            tips["training_crop_button"]
        )
        self.flow_diagram_widget.inference_crop_button.setToolTip(
            tips["denoising_crop_button"]
        )
        self.flow_diagram_widget.preprocess_button.setToolTip(tips["preprocess_button"])
        self.flow_diagram_widget.denoise_button.setToolTip(tips["denoise_button"])
        self.flow_diagram_widget.postprocess_button.setToolTip(
            tips["postprocess_button"]
        )

        # Action buttons on the right side of the navbar
        self.start_button = QPushButton(
            "Start", icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.start_button.setIconSize(QSize(30, 30))
        self.start_button.setMinimumHeight(30)
        self.start_button.setMinimumWidth(70)
        self.start_button.setToolTip(tips["start_button"])
        self.navbar_layout.addWidget(self.start_button)

        self.processing_job_runner = DenoiseJobRunner(
            self, self.threadpool, self.start_button
        )

        self.napari_button = QPushButton("View images")
        self.napari_button.setMinimumHeight(30)
        self.napari_button.setIconSize(QSize(30, 30))
        self.napari_button.clicked.connect(self.open_images_with_napari)
        if sys.platform != 'darwin':
            self.napari_button.setIcon(
                QApplication.style().standardIcon(QStyle.SP_CommandLink)
            )
        self.napari_button.setToolTip(self.tooltips.json["main_page"]["napari_button"])
        if self.napari_viewer is not None:
            self.napari_button.setVisible(False)
        self.navbar_layout.addWidget(self.napari_button)

        self.activity_button = QPushButton("Activity")
        self.activity_button.setMinimumHeight(30)
        self.activity_button.clicked.connect(
            lambda: self.activity_dock.setHidden(not self.activity_dock.isHidden())
        )
        self.activity_button.setIcon(
            QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        )
        self.activity_button.setIconSize(QSize(30, 30))
        self.activity_button.setToolTip(
            self.tooltips.json["main_page"]["activity_button"]
        )
        self.navbar_layout.addWidget(self.activity_button)

        # Force all action buttons to the same fixed height so that
        # text-only buttons (e.g. View images on macOS) match icon buttons,
        # without stretching to the full flow-diagram row height.
        btn_height = self.start_button.sizeHint().height()
        for btn in (self.start_button, self.napari_button, self.activity_button):
            btn.setFixedHeight(btn_height)

        self.main_layout.addLayout(self.navbar_layout, 0)

        # TabWidget (fills remaining space).  Cap minimum height so the
        # Activity dock can grow — tab content is in scroll areas and
        # handles smaller sizes gracefully.
        self.tabwidget = QTabWidget(self)
        self.tabwidget.setMinimumHeight(200)
        self.tabwidget.currentChanged.connect(self.on_tab_change)
        for key, value in self.tabs.items():
            self.tabwidget.addTab(value, key)

        if self.napari_viewer is not None:
            files_idx = self.tabwidget.indexOf(self.tabs["File(s)"])
            self.tabwidget.setTabText(files_idx, "Sources")

        self.main_layout.addWidget(self.tabwidget, 1)

        # Width threshold below which the flow diagram auto-hides
        self._flow_diagram_min_width = 0

        self.overlay = Overlay(self)
        self.overlay.hide()

        # Set layout for the main page widget
        self.setLayout(self.main_layout)

        self.tabs["Dimensions"].dimensions = None
        self.tabs["Training Crop"].images = []
        self.tabs["Denoising Crop"].images = []

    def on_tab_change(self, current_tab_index):
        """Highlight the corresponding flow diagram button when a tab is selected.

        Parameters
        ----------
        current_tab_index : int
            Index of the newly selected tab.
        """
        self.flow_diagram_widget.highlight_button(
            self.tabwidget.tabText(current_tab_index)
        )

    def enable_disable_a_tab(self, class_of_tab_to_disable, enable):
        """Enable or disable a tab identified by its widget class.

        Parameters
        ----------
        class_of_tab_to_disable : type
            The class of the tab widget to enable or disable.
        enable : bool
            Whether to enable (True) or disable (False) the tab.
        """
        for index, tab in enumerate(self.tabs.values()):
            if tab.__class__ == class_of_tab_to_disable:
                self.tabwidget.setTabEnabled(index, enable)
                return

    def resizeEvent(self, event):
        """Resize the overlay and toggle flow diagram visibility.

        The flow diagram auto-hides when the window is too narrow for it
        to display properly. The tab widget always provides equivalent
        navigation.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        self.overlay.resize(event.size())

        # Compute the flow diagram's natural width on first resize
        if self._flow_diagram_min_width == 0:
            self._flow_diagram_min_width = (
                self.flow_diagram_widget.sizeHint().width() + 20
            )

        # Auto-hide the flow diagram when the window is too narrow
        # Only toggle when visibility actually changes to avoid expensive
        # layout recalculations on every resize event.
        should_show = event.size().width() >= self._flow_diagram_min_width
        if self.flow_diagram_widget.isVisible() != should_show:
            self.flow_diagram_widget.setVisible(should_show)

        event.accept()

    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        """Accept drag events that contain file URLs.

        Parameters
        ----------
        e : QDragEnterEvent
            The drag enter event.
        """
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        """Accept drag move events that contain file URLs.

        Parameters
        ----------
        e : QDragMoveEvent
            The drag move event.
        """
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """Handle file drop events by adding dropped files to the data model.

        Switches to the File(s) tab and loads the dropped file paths into
        the data model for processing.

        Parameters
        ----------
        e : QDropEvent
            The drop event containing file URLs.
        """
        if e.mimeData().hasUrls():
            e.setDropAction(Qt.CopyAction)
            e.accept()
            fnames = []

            for url in e.mimeData().urls():
                fnames.append(str(url.toLocalFile()))

            self.tabwidget.setCurrentIndex(self.tabwidget.indexOf(self.tabs["File(s)"]))
            self.data_model.add_filepaths(fnames)
        else:
            e.ignore()

    def load_sample_image(self, sample):
        """Load a sample/example image into the application.

        Parameters
        ----------
        sample : aydin.io.datasets.ExampleDataset
            Sample dataset object that provides a path to a downloadable image.
        """
        try:
            self.data_model.add_filepaths([sample.get_path()])
            self.tabwidget.setCurrentIndex(self.tabwidget.indexOf(self.tabs["File(s)"]))
        except Exception:
            # Download failed:
            # printing stack trace
            aprint("Failed to download or open file!")
            traceback.print_exception(*sys.exc_info())

    def _add_napari_layers(self):
        """Import selected napari image layers (or all if none selected)."""
        if self.napari_viewer is None:
            return

        import napari

        from aydin.napari_plugin._axes_utils import detect_axes_from_napari_layer

        # Prefer selected layers, fall back to all
        selected = [
            layer
            for layer in self.napari_viewer.layers.selection
            if isinstance(layer, napari.layers.Image)
        ]
        image_layers = selected or [
            layer
            for layer in self.napari_viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

        if not image_layers:
            self.status_bar.showMessage('No image layers in napari viewer')
            return

        arrays_dict = {}
        for layer in image_layers:
            if layer.name not in [img.filename for img in self.data_model.images]:
                metadata = detect_axes_from_napari_layer(layer, self.napari_viewer)
                arrays_dict[layer.name] = (layer.data.copy(), metadata)

        if arrays_dict:
            self.data_model.add_arrays(arrays_dict)
            self.tabwidget.setCurrentIndex(
                self.tabwidget.indexOf(self.tabs["Image(s)"])
            )
        else:
            self.status_bar.showMessage('All napari layers already loaded')

    def handle_use_same_crop_state_changed(self):
        """Toggle the Denoising Crop tab based on the Training Crop checkbox.

        When 'Use same cropping for denoising' is checked, the Denoising
        Crop tab is disabled since training crop settings are reused.
        """
        self.enable_disable_a_tab(
            self.tabs["Denoising Crop"].__class__,
            not self.tabs["Training Crop"].use_same_crop_checkbox.isChecked(),
        )

    def toggle_basic_advanced_mode(self):
        """Toggle between basic and advanced mode in the GUI.

        In basic mode, only commonly used options and algorithms are shown.
        In advanced mode, all options and algorithms are available.
        """
        # make calls to toggle the GUI at lower levels
        self.tabs["Pre/Post-Processing"].set_advanced_enabled(
            self.parent.advancedModeButton.isEnabled()
        )
        self.tabs["Denoise"].set_advanced_enabled(
            self.parent.advancedModeButton.isEnabled()
        )

        # swap the enabled state of  `basic` and `advanced` menu items
        self.parent.basicModeButton.setEnabled(
            not self.parent.basicModeButton.isEnabled()
        )
        self.parent.advancedModeButton.setEnabled(
            not self.parent.advancedModeButton.isEnabled()
        )

    def _toggle_spatial_features(self):
        """Update spatial feature availability based on crop slider state.

        When the training crop sliders are adjusted away from their full
        range, spatial features are disabled since they would be misleading
        on a cropped region.
        """
        disable_spatial_features = self.tabs["Training Crop"].disable_spatial_features()
        if disable_spatial_features != self.disable_spatial_features:
            self.tabs["Denoise"].disable_spatial_features = disable_spatial_features
            # `not` is needed to just refresh Denoise tab
            self.tabs["Denoise"].set_advanced_enabled(
                not self.parent.advancedModeButton.isEnabled()
            )
            self.disable_spatial_features = disable_spatial_features

    def add_activity_dockable(self):
        """Create and add the activity log dock widget to the main window."""
        self.activity_dock.setHidden(True)
        self.parent.addDockWidget(Qt.BottomDockWidgetArea, self.activity_dock)
        self.activity_widget = ActivityWidget(self)
        self.activity_dock.setWidget(self.activity_widget)

    def open_images_with_napari(self):
        """Open a napari viewer to display input and denoised images.

        When launched from napari (``self.napari_viewer`` is set), only adds
        denoised results to the existing viewer.  Otherwise, creates a new
        napari viewer showing training images, inference images, and results
        in a grid layout.
        """
        if self.napari_viewer is not None:
            # Only send denoised results to the existing napari viewer
            try:
                image_names = self.processing_job_runner.image_names
                for ind, image_array in enumerate(
                    self.processing_job_runner.result_image_arrays()
                ):
                    src_name = image_names[ind] if ind < len(image_names) else str(ind)
                    self.napari_viewer.add_image(
                        image_array, name=f"{src_name}_denoised"
                    )
            except RuntimeError:
                # Viewer was closed — fall through to create a new one
                self.napari_viewer = None
                self.napari_button.setVisible(True)
            else:
                return

        training_images = self.tabs["Training Crop"].images
        inference_images = (
            self.tabs["Denoising Crop"].images
            if self.tabwidget.isTabEnabled(
                self.tabwidget.indexOf(self.tabs["Denoising Crop"])
            )
            else self.tabs["Training Crop"].images
        )

        import napari

        viewer = napari.Viewer()
        viewer.show()

        if not self.tabwidget.isTabEnabled(
            self.tabwidget.indexOf(self.tabs["Denoising Crop"])
        ):
            # Training&Inference images
            for ind, image_array in enumerate(training_images):
                viewer.add_image(image_array, name=f"{ind}-training&inference_input")
        else:
            # Training images
            for ind, image_array in enumerate(training_images):
                viewer.add_image(image_array, name=f"{ind}-training_input")

            # Inference images
            for ind, image_array in enumerate(inference_images):
                viewer.add_image(image_array, name=f"{ind}-inference_input")

        # Result images
        image_names = self.processing_job_runner.image_names
        for ind, image_array in enumerate(
            self.processing_job_runner.result_image_arrays()
        ):
            src_name = image_names[ind] if ind < len(image_names) else str(ind)
            viewer.add_image(image_array, name=f"{src_name}_denoised")

        viewer.grid.enabled = True

    def save_options_json(self, path=None):
        """Save current denoising options as a JSON file.

        Parameters
        ----------
        path : str, optional
            Specific path to save the JSON file. If None, saves alongside
            each image marked for denoising.
        """
        args_dict = self.tabs["Denoise"].lower_level_args
        args_dict["processing"] = self.tabs["Pre/Post-Processing"].transforms

        if path is None:
            image_paths = [
                get_options_json_path(i.filepath)
                for i in self.data_model.images_to_denoise
            ]
        else:
            image_paths = [path]

        for path in image_paths:
            save_any_json(args_dict, path)

    def load_pretrained_model(self):
        """Open a file dialog to load pretrained model files into the Denoise tab."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open File(s)", "", "All Files (*)", options=options
        )

        if files:
            self.tabs["Denoise"].load_pretrained_model(pretrained_model_files=files)

    def filestab_changed(self):
        """Notify the File(s) tab that the data model has changed."""
        self.tabs["File(s)"].on_data_model_update()

    def imagestab_changed(self):
        """Notify the Image(s) tab that the data model has changed."""
        self.tabs["Image(s)"].on_data_model_update()

    def dimensionstab_changed(self):
        """Notify the Dimensions tab that the data model has changed."""
        self.tabs["Dimensions"].on_data_model_update()

    def croppingtabs_changed(self):
        """Notify the cropping tabs that the data model changed."""
        self.tabs["Training Crop"].on_data_model_update()
        self.tabs["Denoising Crop"].on_data_model_update()
