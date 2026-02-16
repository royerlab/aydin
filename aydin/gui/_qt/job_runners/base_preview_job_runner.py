"""Base class for preview job runners with shared thread management."""

from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QStyle, QWidget

from aydin.gui._qt.job_runners.worker import Worker
from aydin.util.log.log import Log, aprint


class BasePreviewJobRunner(QWidget):
    """Base class for preview job runners.

    Provides shared ``__init__``, ``progress_fn``, ``error_fn``,
    ``thread_complete``, and ``_launch_worker`` methods. Subclasses
    must implement ``start_func`` and ``prep_and_run``.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    threadpool : QThreadPool
        Thread pool for executing background workers.
    main_page : MainPage
        Reference to the main page widget for accessing shared resources.
    button_text : str
        Label for the preview button.
    button_width : int
        Fixed width of the preview button in pixels.
    """

    def __init__(self, parent, threadpool, main_page, button_text, button_width):
        super().__init__(parent)
        self.parent = parent
        self.threadpool = threadpool
        self.main_page = main_page

        self.result_images = []
        self.preprocessed = []
        self.postprocessed = []

        self.widget_layout = QHBoxLayout()
        self.start_button = QPushButton(
            button_text, icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.start_button.setFixedWidth(button_width)
        self.start_button.clicked.connect(self.prep_and_run)
        self.widget_layout.addWidget(self.start_button)

        self.setLayout(self.widget_layout)

    def start_func(self, progress_callback):
        """Apply transforms to images. Must be overridden by subclasses."""
        raise NotImplementedError

    def progress_fn(self, log_str):
        """Append progress text to the activity log.

        Parameters
        ----------
        log_str : str
            Text to append.
        """
        self.main_page.activity_widget.infoTextBox.insertPlainText(log_str)

    def error_fn(self, error_tuple):
        """Handle worker thread errors by logging and re-enabling the UI.

        Parameters
        ----------
        error_tuple : tuple
            Tuple of (exctype, value, formatted_traceback).
        """
        exctype, value, tb_str = error_tuple
        aprint(f"Preview failed: {value}\n{tb_str}")
        self.main_page.activity_widget.infoTextBox.insertPlainText(
            f"ERROR: {value}\n{tb_str}"
        )
        self.start_button.setEnabled(True)

    def thread_complete(self):
        """Re-enable the preview button and open napari with the results."""
        self.start_button.setEnabled(True)

        if self.preprocessed != [] and self.postprocessed != []:
            import napari

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

    def _launch_worker(self):
        """Create and start the background worker thread."""
        Log.gui_statusbar = self.main_page.parent.status_bar

        # Show activity widget
        self.main_page.activity_dock.setHidden(False)

        worker = Worker(self.start_func)

        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        worker.signals.error.connect(self.error_fn)

        self.main_page.activity_widget.clear_activity()

        self.result_images = []

        self.start_button.setEnabled(False)

        self.threadpool.start(worker)

    def prep_and_run(self):
        """Gather settings and launch the worker. Must be overridden."""
        raise NotImplementedError
