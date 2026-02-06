"""Job runner for previewing a single transform in a background thread."""

import napari
from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QStyle, QWidget

from aydin.gui._qt.job_runners.worker import Worker
from aydin.util.log.log import Log, aprint


class PreviewJobRunner(QWidget):
    """Runs a single transform preview in a background thread.

    Applies a transform's preprocess and postprocess steps to the training
    images and displays the results in a napari viewer.

    Parameters
    ----------
    parent : TransformsTabItem
        The parent transform tab item widget.
    threadpool : QThreadPool
        Thread pool for executing background workers.
    """

    def __init__(self, parent, threadpool):
        super(PreviewJobRunner, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool

        self.result_images = []
        self.preprocessed = []
        self.postprocessed = []

        self.widget_layout = QHBoxLayout()
        self.start_button = QPushButton(
            "Preview", icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.start_button.setFixedWidth(140)
        self.start_button.clicked.connect(self.prep_and_run)
        self.widget_layout.addWidget(self.start_button)

        self.setLayout(self.widget_layout)

    def start_func(self, progress_callback):
        """Apply the transform's pre- and post-processing to each image.

        Parameters
        ----------
        progress_callback : Signal
            Qt signal for reporting progress text.
        """
        Log.gui_callback = progress_callback

        for image in self.images:
            transform_instance = self.transform_class(
                **self._prepare_params_dict["kwargs"]
            )

            try:
                self.preprocessed.append(transform_instance.preprocess(image))
                self.postprocessed.append(
                    transform_instance.postprocess(self.preprocessed[-1])
                )
            except BaseException as e:
                error_message = str(e).replace('\n', ', ')
                aprint(
                    f"Preprocessing failed for {transform_instance} with: {error_message} "
                )

        Log.gui_callback = None

    def progress_fn(self, log_str):
        """Append progress text to the activity log.

        Parameters
        ----------
        log_str : str
            Text to append.
        """
        self.parent.parent.parent.parent.activity_widget.infoTextBox.insertPlainText(
            log_str
        )

    def thread_complete(self):
        """Re-enable the preview button and open napari with the results."""
        self.start_button.setEnabled(True)

        if self.preprocessed != [] and self.postprocessed != []:
            viewer = napari.Viewer()
            for image, preprocessed, postprocessed in zip(
                self.images, self.preprocessed, self.postprocessed
            ):
                viewer.add_image(image, name="image")
                viewer.add_image(preprocessed, name="preprocessed")
                viewer.add_image(postprocessed, name="postprocessed")

            viewer.grid.enabled = True
            viewer.grid.shape = (len(self.images), 3)
            viewer.show()

        self.preprocessed = []
        self.postprocessed = []

    def prep_and_run(self):
        """Gather images and transform settings, then launch the preview worker."""
        # Get images and their related data
        self.images = self.parent.parent.parent.parent.tabs["Training Crop"].images

        if len(self.images) == 0:
            aprint("Preview cannot be started with no image")
            return

        self.transform_class = self.parent.transform_class

        # We trick parent.preprocess_checkbox to get params_dict whether related processing is enabled or not
        preprocess_checkbox_value = self.parent.preprocess_checkbox.isChecked()
        self.parent.preprocess_checkbox.setChecked(True)
        self._prepare_params_dict = self.parent.params_dict
        self.parent.preprocess_checkbox.setChecked(preprocess_checkbox_value)

        Log.gui_statusbar = self.parent.parent.parent.parent.parent.statusBar

        # Show activity widget
        self.parent.parent.parent.parent.activity_dock.setHidden(False)

        # Pass the function to execute
        worker = Worker(
            self.start_func
        )  # Any other args, kwargs are passed to the run function

        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        self.parent.parent.parent.parent.activity_widget.clear_activity()

        self.result_images = []

        self.start_button.setEnabled(False)

        # Execute
        self.threadpool.start(worker)
