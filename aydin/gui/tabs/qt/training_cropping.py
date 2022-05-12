import math
from qtpy.QtWidgets import QCheckBox

from aydin.gui.tabs.qt.base_cropping import BaseCroppingTab
from aydin.util.crop.super_fast_rep_crop import super_fast_representative_crop


class TrainingCroppingTab(BaseCroppingTab):
    """
    Cropping Image for training & auto-tuning

    Use the sliders to select a region of the image to define the cropping
    region for training or auto-tuning. Aydin automatically suggests a crop
    based on the image content.

    <moreless>

    It is often advised not to use the entire image for training or
    calibration You may want to focus the attention of the algorithm to a
    specific region with structures of interest and exclude excessive dark
    background. Or simply reduce the time required for training or
    auto-tuning. In the case of neural networks (CNN, NN) and dictionary
    based methods more data does help, methods using gradient boosting less
    so (N2S-FGR-cb/lgbm), and simpler methods such as low-pass filtering (
    Butterworth, Gaussian, ...) need much less data (but ideally highly
    contrasted and detail rich regions) to work. if the results are
    unsatisfactory, try extending the cropped region to encompass more of the
    image (within reason).

    <split>
    """

    def __init__(self, parent):
        super(TrainingCroppingTab, self).__init__(parent)
        self.parent = parent

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
            response = super_fast_representative_crop(
                image,
                mode='contrast',
                crop_size=2_000_000,
                search_mode='random' if image.size > 500_000_000 else 'systematic',
                random_search_mode_num_crops=1024,
                return_slice=True,
                timeout_in_seconds=1.5,
            )

            if type(response) == tuple:
                best_slice = response[1]

                t_slice = best_slice[images[0][2].axes.find("T")]
                z_slice = best_slice[images[0][2].axes.find("Z")]
                y_slice = best_slice[images[0][2].axes.find("Y")]
                x_slice = best_slice[images[0][2].axes.find("X")]
                self.t_crop_slider.slider.setValues((t_slice.start, t_slice.stop))
                self.z_crop_slider.slider.setValues((z_slice.start, z_slice.stop))
                self.y_crop_slider.slider.setValues((y_slice.start, y_slice.stop))
                self.x_crop_slider.slider.setValues((x_slice.start, x_slice.stop))

    def update_summary(self):
        super().update_summary()

        self.parent._toggle_spatial_features()

        if math.prod(self.cropped_image[0].shape) > 50_000_000:
            self.size_warning_label.setText(
                "Training might take a long time, please consider to reduce the size of crop as much as possible"
            )
            self.size_warning_label.setStyleSheet(
                "QLabel {font-weight: bold; color: red;}"
            )
        else:
            self.size_warning_label.setText("")

    def disable_spatial_features(self):
        return (
            (
                not self.x_crop_slider.isHidden()
                and self.x_crop_slider.slider.range()
                != self.x_crop_slider.slider.values()
            )
            or (
                not self.y_crop_slider.isHidden()
                and self.y_crop_slider.slider.range()
                != self.y_crop_slider.slider.values()
            )
            or (
                not self.z_crop_slider.isHidden()
                and self.z_crop_slider.slider.range()
                != self.z_crop_slider.slider.values()
            )
            or (
                not self.t_crop_slider.isHidden()
                and self.t_crop_slider.slider.range()
                != self.t_crop_slider.slider.values()
            )
        )
