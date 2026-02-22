"""Job runner for previewing a single transform in a background thread."""

from aydin.gui._qt.job_runners.base_preview_job_runner import BasePreviewJobRunner
from aydin.util.log.log import Log, aprint


class PreviewJobRunner(BasePreviewJobRunner):
    """Runs a single transform preview in a background thread.

    Applies a transform's preprocess and postprocess steps to the training
    images and displays the results in a napari viewer.

    Parameters
    ----------
    parent : TransformsTabItem
        The parent transform tab item widget.
    threadpool : QThreadPool
        Thread pool for executing background workers.
    main_page : MainPage
        Reference to the main page widget for accessing shared resources.
    """

    def __init__(self, parent, threadpool, main_page=None):
        """Initialize the preview job runner with a Preview button.

        Parameters
        ----------
        parent : TransformsTabItem
            The parent transform tab item widget.
        threadpool : QThreadPool
            Thread pool for executing background workers.
        main_page : MainPage, optional
            Reference to the main page widget for accessing shared resources.
        """
        super().__init__(
            parent, threadpool, main_page, button_text="Preview", button_width=140
        )

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
            except Exception as e:
                error_message = str(e).replace('\n', ', ')
                aprint(
                    f"Preprocessing failed for"
                    f" {transform_instance}"
                    f" with: {error_message} "
                )

        Log.gui_callback = None

    def prep_and_run(self):
        """Gather images and transform settings, then launch the preview worker."""
        # Get images and their related data
        self.images = self.main_page.tabs["Training Crop"].images

        if len(self.images) == 0:
            aprint("Preview cannot be started with no image")
            return

        self.transform_class = self.parent.transform_class

        # We trick parent.preprocess_checkbox to get params_dict
        # whether related processing is enabled or not
        preprocess_checkbox_value = self.parent.preprocess_checkbox.isChecked()
        self.parent.preprocess_checkbox.setChecked(True)
        self._prepare_params_dict = self.parent.params_dict
        self.parent.preprocess_checkbox.setChecked(preprocess_checkbox_value)

        self._launch_worker()
