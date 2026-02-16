"""Job runner for previewing all enabled transforms in a background thread."""

from aydin.gui._qt.job_runners.base_preview_job_runner import BasePreviewJobRunner
from aydin.it.fgr import ImageTranslatorFGR
from aydin.util.log.log import Log, aprint


class PreviewAllJobRunner(BasePreviewJobRunner):
    """Runs a combined preview of all enabled transforms in a background thread.

    Applies all selected transforms sequentially to the training images and
    displays the combined pre- and post-processed results in a napari viewer.

    Parameters
    ----------
    parent : ProcessingTab
        The parent processing tab widget.
    threadpool : QThreadPool
        Thread pool for executing background workers.
    main_page : MainPage
        Reference to the main page widget for accessing shared resources.
    """

    def __init__(self, parent, threadpool, main_page=None):
        """Initialize the preview-all job runner with a Preview All button.

        Parameters
        ----------
        parent : ProcessingTab
            The parent processing tab widget.
        threadpool : QThreadPool
            Thread pool for executing background workers.
        main_page : MainPage, optional
            Reference to the main page widget for accessing shared resources.
        """
        super().__init__(
            parent, threadpool, main_page, button_text="Preview all", button_width=180
        )

    def start_func(self, progress_callback):
        """Apply all enabled transforms to each image in sequence.

        Parameters
        ----------
        progress_callback : Signal
            Qt signal for reporting progress text.
        """
        Log.gui_callback = progress_callback

        for image in self.images:

            it = ImageTranslatorFGR()

            for transform in self.parent.transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                it.add_transform(transform_class(**transform_kwargs))

            self.preprocessed.append(it.transform_preprocess_image(image))
            self.postprocessed.append(
                it.transform_postprocess_image(self.preprocessed[-1])
            )

        Log.gui_callback = None

    def prep_and_run(self):
        """Gather images and transform settings, then launch the preview-all worker."""
        # Get images and their related data
        self.images = self.main_page.tabs["Training Crop"].images
        if len(self.images) == 0:
            aprint("Preview All cannot be started with no image")
            return

        self._launch_worker()
