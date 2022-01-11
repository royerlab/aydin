from qtpy.QtWidgets import QWidget, QHBoxLayout

from aydin.gui._qt.output_wrapper import OutputWrapper
from aydin.gui._qt.job_runners.worker import Worker
from aydin.io.io import imwrite
from aydin.io.utils import (
    get_output_image_path,
    get_options_json_path,
    get_save_model_path,
)
from aydin.restoration.denoise.util.denoise_utils import get_denoiser_class_instance
from aydin.util.log.log import Log, lprint


class DenoiseJobRunner(QWidget):
    def __init__(self, parent, threadpool, start_button):
        super(QWidget, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool
        self.start_button = start_button

        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.progress_fn)
        stderr = OutputWrapper(self, False)
        stderr.outputWritten.connect(self.progress_fn)

        self.result_images = []
        self.early_stopped = False

        self.widget_layout = QHBoxLayout()

        self.start_button.clicked.connect(self.prep_and_run)

        self.setLayout(self.widget_layout)

    def stop_running(self):
        self.early_stopped = True
        self.denoiser.stop_running()

    def start_func(self, progress_callback):
        Log.gui_callback = progress_callback

        results = []

        for training_image, inference_image, image_path in zip(
            self.training_images, self.inference_images, self.image_paths
        ):
            self.denoiser.train(
                training_image, batch_axes=self.batch_axes, chan_axes=self.channel_axes
            )

            if self.denoiser.it:
                denoised = self.denoiser.denoise(
                    inference_image,
                    batch_axes=self.batch_axes,
                    chan_axes=self.channel_axes,
                )
                results.append(denoised)

                output_path, file_counter = get_output_image_path(image_path)

                imwrite(denoised, output_path)
                if self.save_options_json:
                    json_path = get_options_json_path(
                        image_path, passed_counter=file_counter
                    )
                    self.parent.save_options_json(json_path)
                    lprint(f"DONE, options json written in {json_path}")

                if self.save_model:
                    model_path = get_save_model_path(
                        image_path, passed_counter=file_counter
                    )
                    self.denoiser.save_model(model_path)
                    lprint(f"DONE, trained model written in {model_path}")

                lprint(f"DONE, results written in {output_path}")
            else:
                self.early_stopped = True
                lprint("DONE, failed to run...")

        Log.gui_callback = None
        return results

    def progress_fn(self, log_str):
        self.parent.activity_widget.infoTextBox.insertPlainText(log_str)

    def result_callback(self, results):
        self.result_images += results

    def thread_complete(self):
        self.start_button.setEnabled(True)

        # Turn the overlay off
        self.parent.overlay.hide()

        # Open napari with input and output images
        if not self.early_stopped:
            self.parent.open_images_with_napari()

    def prep_and_run(self):
        # Get images and their related data
        self.image_paths = [i[5] for i in self.parent.data_model.images_to_denoise]
        if len(self.image_paths) == 0:
            lprint("Aydin cannot be started with no image")
            return

        self.training_images = self.parent.tabs["Training Crop"].images
        self.inference_images = (
            self.parent.tabs["Denoising Crop"].images
            if self.parent.tabwidget.isTabEnabled(
                self.parent.tabwidget.indexOf(self.parent.tabs["Denoising Crop"])
            )
            else self.parent.tabs["Training Crop"].images
        )
        self.batch_axes = self.parent.tabs["Dimensions"].batch_axes
        self.channel_axes = self.parent.tabs["Dimensions"].channel_axes
        self.denoise_backend = self.parent.tabs["Denoise"].selected_backend

        try:
            self.it_transforms = self.parent.tabs["Pre/Post-Processing"].transforms
            self.lower_level_args = self.parent.tabs["Denoise"].lower_level_args
        except Exception:
            self.parent.status_bar.showMessage(
                "There is a mistake with given parameter values..."
            )
            return

        self.save_options_json = self.parent.tabs[
            "Denoise"
        ].current_backend_widget.save_json_checkbox.isChecked()

        self.save_model = self.parent.tabs[
            "Denoise"
        ].current_backend_widget.save_model_checkbox.isChecked()

        # Initialize required restoration instances
        self.denoiser = get_denoiser_class_instance(
            variant=self.denoise_backend,
            lower_level_args=self.lower_level_args,
            it_transforms=self.it_transforms,
        )

        Log.gui_statusbar = self.parent.parent.statusBar

        # Show activity widget
        self.parent.activity_dock.setHidden(False)

        # Turn the overlay on
        self.parent.overlay.show()

        # Pass the function to execute
        worker = Worker(
            self.start_func
        )  # Any other args, kwargs are passed to the run function

        worker.signals.result.connect(self.result_callback)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        self.parent.activity_widget.clear_activity()

        self.result_images = []

        self.start_button.setEnabled(False)

        # Execute
        self.threadpool.start(worker)

    def result_image_arrays(self):
        return self.result_images
