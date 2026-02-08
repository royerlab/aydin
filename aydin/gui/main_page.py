"""Central widget for Aydin Studio containing the tabbed workflow interface."""

import sys
import traceback

import napari
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
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

    def __init__(self, parent, threadpool, status_bar):
        super(MainPage, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool
        self.status_bar = status_bar

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

        # MainPage layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)

        # navbar
        self.navbar_layout = QHBoxLayout()
        self.navbar_layout.setAlignment(Qt.AlignCenter)
        self.navbar_layout.setContentsMargins(0, 0, 0, 0)
        self.navbar_layout.setSpacing(5)

        self.add_activity_dockable()

        self.navbar_layout_center = QHBoxLayout()

        self.flow_diagram_widget = QProgramFlowDiagramWidget(self)
        self.navbar_layout_center.addWidget(self.flow_diagram_widget)

        self.flow_diagram_widget.add_files_button.clicked.connect(
            self.tabs["File(s)"].openFileNamesDialog
        )
        self.flow_diagram_widget.add_files_button.setIcon(
            QApplication.style().standardIcon(QStyle.SP_DirIcon)
        )
        self.flow_diagram_widget.add_files_button.setToolTip(
            self.tooltips.json["main_page"]["open_file_button"]
        )

        self.flow_diagram_widget.files_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(1)
        )
        self.flow_diagram_widget.images_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(2)
        )
        self.flow_diagram_widget.dimensions_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(3)
        )
        self.flow_diagram_widget.training_crop_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(4)
        )
        self.flow_diagram_widget.inference_crop_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(5)
        )
        self.flow_diagram_widget.preprocess_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(6)
        )
        self.flow_diagram_widget.postprocess_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(6)
        )
        self.flow_diagram_widget.denoise_button.clicked.connect(
            lambda: self.tabwidget.setCurrentIndex(7)
        )

        self.start_button = QPushButton(
            "Start", icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.start_button.setIconSize(QSize(30, 30))
        self.start_button.setMinimumHeight(30)
        self.start_button.setMinimumWidth(70)
        self.start_button.setStyleSheet("QPushButton {font-weight: bold;}")
        self.navbar_layout_center.addWidget(self.start_button)

        self.processing_job_runner = DenoiseJobRunner(
            self, self.threadpool, self.start_button
        )

        self.napari_button = QPushButton("View images")
        self.napari_button.setStyleSheet("QPushButton {font-weight: bold;}")
        self.napari_button.setMinimumHeight(30)
        self.napari_button.clicked.connect(self.open_images_with_napari)
        self.napari_button.setIcon(
            QApplication.style().standardIcon(QStyle.SP_CommandLink)
        )
        self.napari_button.setIconSize(QSize(30, 30))
        self.napari_button.setToolTip(self.tooltips.json["main_page"]["napari_button"])
        self.navbar_layout_center.addWidget(self.napari_button)

        self.navbar_layout_center.setAlignment(Qt.AlignCenter)
        self.navbar_layout.addLayout(self.navbar_layout_center)

        self.navbar_layout_right = QHBoxLayout()
        self.activity_button = QPushButton("Activity")
        self.activity_button.setStyleSheet("QPushButton {font-weight: bold;}")
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
        self.navbar_layout_center.addWidget(self.activity_button)

        self.navbar_layout_right.setAlignment(Qt.AlignRight)
        self.navbar_layout.addLayout(self.navbar_layout_right)

        self.navbar_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addLayout(self.navbar_layout)

        # TabWidget
        self.tabwidget = QTabWidget(self)
        self.tabwidget.currentChanged.connect(self.onTabChange)
        for key, value in self.tabs.items():
            self.tabwidget.addTab(value, key)

        self.main_layout.addWidget(self.tabwidget)

        self.overlay = Overlay(self)
        self.overlay.hide()

        # Set layout for the main page widget
        self.setLayout(self.main_layout)

        self.tabs["Dimensions"].dimensions = None
        self.tabs["Training Crop"].images = []
        self.tabs["Denoising Crop"].images = []

    def onTabChange(self, current_tab_index):
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
        """Resize the overlay to match the widget size.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        self.overlay.resize(event.size())
        event.accept()

    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        """Accept drag events that contain file URLs.

        Parameters
        ----------
        e : QDragEnterEvent
            The drag enter event.
        """
        if e.mimeData().hasUrls:
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
        if e.mimeData().hasUrls:
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
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            fnames = []

            for url in e.mimeData().urls():
                fnames.append(str(url.toLocalFile()))

            self.tabwidget.setCurrentIndex(1)
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
            self.tabwidget.setCurrentIndex(1)
        except Exception:
            # Download failed:
            # printing stack trace
            aprint("Failed to download or open file!")
            traceback.print_exception(*sys.exc_info())

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
            self.tabs["Denoise"].set_advanced_enabled(
                not self.parent.advancedModeButton.isEnabled()  # `not` is needed to just refresh Denoise tab
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

        Shows training images, inference images (if separate from training),
        and any denoised result images in a grid layout.
        """
        training_images = self.tabs["Training Crop"].images
        inference_images = (
            self.tabs["Denoising Crop"].images
            if self.tabwidget.isTabEnabled(
                self.tabwidget.indexOf(self.tabs["Denoising Crop"])
            )
            else self.tabs["Training Crop"].images
        )

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
        for ind, image_array in enumerate(
            self.processing_job_runner.result_image_arrays()
        ):
            viewer.add_image(image_array, name=f"{ind}-denoised_output")

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
                get_options_json_path(i[4]) for i in self.data_model.images_to_denoise
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
        """Notify the Training Crop and Denoising Crop tabs that the data model has changed."""
        self.tabs["Training Crop"].on_data_model_update()
        self.tabs["Denoising Crop"].on_data_model_update()
