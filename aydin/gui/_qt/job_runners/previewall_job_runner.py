import napari
from qtpy.QtWidgets import QWidget, QPushButton, QApplication, QStyle, QHBoxLayout

from aydin.gui._qt.job_runners.worker import Worker
from aydin.it.fgr import ImageTranslatorFGR
from aydin.util.log.log import Log, lprint


class PreviewAllJobRunner(QWidget):
    def __init__(self, parent, threadpool):
        super(QWidget, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool

        self.result_images = []
        self.preprocessed = []
        self.postprocessed = []

        self.widget_layout = QHBoxLayout()
        self.start_button = QPushButton(
            "Preview all", icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.start_button.setFixedWidth(180)
        self.start_button.clicked.connect(self.prep_and_run)
        self.widget_layout.addWidget(self.start_button)

        self.setLayout(self.widget_layout)

    def start_func(self, progress_callback):
        Log.gui_callback = progress_callback

        for image in self.images:

            it = ImageTranslatorFGR()

            for transform in self.parent.transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                it.add_transform(transform_class(**transform_kwargs))

            self.preprocessed.append(it._transform_preprocess_image(image))
            self.postprocessed.append(
                it._transform_postprocess_image(self.preprocessed[-1])
            )

        Log.gui_callback = None

    def progress_fn(self, log_str):
        self.parent.parent.activity_widget.infoTextBox.insertPlainText(log_str)

    def thread_complete(self):
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
        # Get images and their related data
        self.images = self.parent.parent.tabs["Training Crop"].images
        if len(self.images) == 0:
            lprint("Preview All cannot be started with no image")
            return

        Log.gui_statusbar = self.parent.parent.parent.statusBar

        # Show activity widget
        self.parent.parent.activity_dock.setHidden(False)

        # Pass the function to execute
        worker = Worker(
            self.start_func
        )  # Any other args, kwargs are passed to the run function

        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        self.parent.parent.activity_widget.clear_activity()

        self.result_images = []

        self.start_button.setEnabled(False)

        # Execute
        self.threadpool.start(worker)
