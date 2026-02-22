"""Widget for displaying a pretrained denoising model in the Denoise tab."""

from qtpy.QtWidgets import QWidget

from aydin.gui._qt.custom_widgets.denoise_tab_common import setup_denoise_tab_layouts


class DenoiseTabPretrainedMethodWidget(QWidget):
    """Widget for a pretrained denoising model loaded from file.

    Displays the model description and save options, but does not expose
    configurable parameters since the model is already trained.

    Parameters
    ----------
    parent : DenoiseTab
        The parent denoise tab widget.
    loaded_it : ImageTranslatorBase
        The loaded pretrained image translator instance.
    """

    def __init__(self, parent, loaded_it):
        """Initialize the pretrained method widget with model info and save options.

        Parameters
        ----------
        parent : DenoiseTab
            The parent denoise tab widget.
        loaded_it : ImageTranslatorBase
            The loaded pretrained image translator instance.
        """
        super(DenoiseTabPretrainedMethodWidget, self).__init__(parent)

        self.parent = parent
        self.loaded_it = loaded_it
        self.name = loaded_it.__class__.__name__
        self.description = (
            f"This is a pretrained model, namely uses the"
            f" image translator: {loaded_it.__class__.__name__},"
            f" will not train anything new but will quickly"
            f" infer on the images of your choice."
        )

        (
            self.main_layout,
            self.tab_method_layout,
            self.right_side_vlayout,
            self.description_scroll,
            self.description_label,
            self.save_json_checkbox,
            self.save_model_checkbox,
        ) = setup_denoise_tab_layouts(self, self.description)

        self.setLayout(self.main_layout)
