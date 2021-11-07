import math
from qtpy.QtWidgets import QCheckBox

from aydin.gui.tabs.qt.base_cropping import BaseCroppingTab
from aydin.util.crop.rep_crop import representative_crop


class TrainingCroppingTab(BaseCroppingTab):
    """
    Cropping Image for training & auto-tuning

    Use the sliders to select a region of the image to define the cropping region for training or auto-tuning. Aydin
    automatically suggests a crop based on the image content.

    <moreless>

    It is often not advised to use the entire image for training or auto-tuning. You may want to focus the attention
    of the algorithm to a specific region, exclude excessive dark background. or simply reduce the time required for
    training or auto-tuning. In the case of neural networks (CNN, NN) and dictionary based methods more data does
    help, methods using gradient boosting less so (N2S-FGR-cb/lgbm), and simpler methods such as low-pass filtering (
    Butterworth, Gaussian, ...) need very data (but ideally highly contrasted and detail rich regions) to work.

    <split>
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.use_same_crop_checkbox = QCheckBox("Use same cropping for denoising")
        self.use_same_crop_checkbox.toggled.connect(
            lambda: parent.handle_use_same_crop_state_changed()
        )

        self.explanation_layout.addStretch()
        self.explanation_layout.addWidget(self.use_same_crop_checkbox, 10)

    @property
    def images(self):
        return super().images

    @images.setter
    def images(self, images):
        super(TrainingCroppingTab, self.__class__).images.fset(self, images)

        if len(images) == 1:
            image = images[0][1]
            response = representative_crop(
                image,
                mode='contrast' if image.size > 1_000_000 else 'sobelmin',
                crop_size=500_000,
                fast_mode=image.size > 2_000_000,
                fast_mode_num_crops=1024,
                return_slice=True,
            )

            if type(response) == tuple:
                best_slice = response[1]

                # TODO: extend this implementation to handle all chosen spatial-temporal dimensions
                y_slice = best_slice[images[0][2].axes.find("Y")]
                x_slice = best_slice[images[0][2].axes.find("X")]
                self.y_crop_slider.slider.setValues((y_slice.start, y_slice.stop))
                self.x_crop_slider.slider.setValues((x_slice.start, x_slice.stop))

    def update_summary(self):

        super().update_summary()

        if math.prod(self.cropped_image[0].shape) > 50_000_000:
            self.size_warning_label.setText(
                "Training might take a long time, please consider to reduce the size of crop as much as possible"
            )
            self.size_warning_label.setStyleSheet(
                "QLabel {font-weight: bold; color: red;}"
            )
        else:
            self.size_warning_label.setText("")
