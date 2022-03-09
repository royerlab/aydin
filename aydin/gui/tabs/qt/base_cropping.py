import numpy
from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel

from aydin.gui._qt.custom_widgets.readmoreless_label import QReadMoreLessLabel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QWidget

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.custom_widgets.qt_slider_with_labels import QRangeSliderWithLabels
from aydin.util.misc.units import human_readable_byte_size


class BaseCroppingTab(QWidget):
    """Use the sliders to select a region of the image to crop."""

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self._image = None
        self._metadata = None
        self._images = []
        self.crop_layer = None

        self.tab_layout = QVBoxLayout()
        self.tab_layout.setAlignment(Qt.AlignTop)

        self.explanation_layout = QHBoxLayout()
        self.explanation_text = QReadMoreLessLabel(self, self.__doc__)
        self.explanation_layout.addWidget(self.explanation_text, 90)
        self.tab_layout.addLayout(self.explanation_layout)

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.viewer_and_sliders_layout = QHBoxLayout()
        self.viewer_and_sliders_layout.setAlignment(Qt.AlignTop)
        # Image
        self.viewer_layout = QVBoxLayout()
        self.viewer_layout.setAlignment(Qt.AlignTop)
        self.viewer_model = ViewerModel()
        self.viewer_qt = QtViewer(self.viewer_model, show_welcome_screen=False)

        self.viewer_layout.addWidget(self.viewer_qt)
        self.viewer_layout.addWidget(self.viewer_qt.dims)
        self.viewer_and_sliders_layout.addLayout(self.viewer_layout, 50)

        self.cropping_selection_layout = QVBoxLayout()
        self.cropping_selection_layout.addWidget(QLabel("Cropping Selection"))
        self.cropping_selection_layout.setAlignment(Qt.AlignTop)
        self.x_crop_slider = QRangeSliderWithLabels(self, label="X")
        self.cropping_selection_layout.addWidget(self.x_crop_slider)
        self.y_crop_slider = QRangeSliderWithLabels(self, label="Y")
        self.cropping_selection_layout.addWidget(self.y_crop_slider)
        self.z_crop_slider = QRangeSliderWithLabels(self, label="Z")
        self.cropping_selection_layout.addWidget(self.z_crop_slider)
        self.t_crop_slider = QRangeSliderWithLabels(self, label="T")
        self.cropping_selection_layout.addWidget(self.t_crop_slider)
        self.summary_layout = QVBoxLayout()
        self.summary_numbers_layout = QVBoxLayout()
        self.summary_numbers_layout.setAlignment(Qt.AlignCenter)
        self.summary_layout.setAlignment(Qt.AlignCenter)
        self.summary_nbvoxels_label = QLabel("Total number of voxels: N/A", self)
        self.summary_numbers_layout.addWidget(self.summary_nbvoxels_label)
        self.summary_nbytes_label = QLabel("Total size in bytes: N/A", self)
        self.summary_numbers_layout.addWidget(self.summary_nbytes_label)
        self.summary_layout.addLayout(self.summary_numbers_layout)

        self.warning_label_layout = QHBoxLayout()
        self.warning_label_layout.setAlignment(Qt.AlignCenter)
        self.size_warning_label = QLabel("", self)
        self.warning_label_layout.addWidget(self.size_warning_label)
        self.summary_layout.addLayout(self.warning_label_layout)
        self.cropping_selection_layout.addLayout(self.summary_layout)
        self.viewer_and_sliders_layout.addLayout(self.cropping_selection_layout, 50)

        self.tab_layout.addLayout(self.viewer_and_sliders_layout)

        self.setLayout(self.tab_layout)

    @property
    def images(self):
        return self.cropped_image if len(self._images) == 1 else self._images

    @images.setter
    def images(self, images):
        if len(images) == 0:
            self._images = []
            self.clear_cropping_tab()
            self.parent.enable_disable_a_tab(self.__class__, False)
        elif len(images) == 1:
            self.clear_cropping_tab()
            self.parent.enable_disable_a_tab(self.__class__, True)
            self._images = images
            self._metadata = images[0][2]
            self.image = images[0][1]
        else:
            self._images = [image[1] for image in images]
            self.clear_cropping_tab()
            self.parent.enable_disable_a_tab(self.__class__, False)

    @property
    def image(self):
        """image property

        Returns
        -------
        Returns the image with downscaled and padded spatiotemporal axes

        """
        return self._image

    @image.setter
    def image(self, image):
        """image property setter.

        Parameters
        ----------
        image

        """
        self._image = image
        self.new_crop = numpy.zeros(
            (
                self._image.shape[self._metadata.axes.find("Y")],
                self._image.shape[self._metadata.axes.find("X")],
            ),
            dtype=numpy.uint8,
        )

        self.initialize_viewer()
        self.update_sliders()

    @property
    def crop_selection_slicing_object(self):
        slider_slice = [slice(None)] * len(self.image.shape)
        for idx, axis in enumerate(self._metadata.axes):
            if axis == "X":
                lower, upper = self.x_crop_slider.slider.values()
                slider_slice[idx] = slice(int(lower), int(upper) + 1, 1)
            elif axis == "Y":
                lower, upper = self.y_crop_slider.slider.values()
                slider_slice[idx] = slice(int(lower), int(upper) + 1, 1)
            elif axis == "Z":
                lower, upper = self.z_crop_slider.slider.values()
                slider_slice[idx] = slice(int(lower), int(upper) + 1, 1)
            elif axis == "T":
                lower, upper = self.t_crop_slider.slider.values()
                slider_slice[idx] = slice(int(lower), int(upper) + 1, 1)

        return tuple(slider_slice)

    @property
    def cropped_image(self):
        """Takes the image apply crop and returns the cropped image.

        Returns
        -------
        Single image in a list

        """
        image = self._image[self.crop_selection_slicing_object]

        return [image]

    @property
    def selection_array(self):
        self.new_crop[...] = 0
        self.new_crop[
            (
                self.crop_selection_slicing_object[self._metadata.axes.find("Y")],
                self.crop_selection_slicing_object[self._metadata.axes.find("X")],
            )
        ] = 100

        return self.new_crop

    def update_sliders(self):
        self.x_crop_slider.slider.setRange(
            (0, self.image.shape[self._metadata.axes.find("X")])
        )
        self.x_crop_slider.slider.setValues(
            (0, self.image.shape[self._metadata.axes.find("X")])
        )
        self.x_crop_slider.upper_limit_label.setText(
            str(self.image.shape[self._metadata.axes.find("X")])
        )

        self.y_crop_slider.slider.setRange(
            (0, self.image.shape[self._metadata.axes.find("Y")])
        )
        self.y_crop_slider.slider.setValues(
            (0, self.image.shape[self._metadata.axes.find("Y")])
        )
        self.y_crop_slider.upper_limit_label.setText(
            str(self.image.shape[self._metadata.axes.find("Y")])
        )

        if "Z" in self._metadata.axes:
            self.z_crop_slider.setHidden(False)
            self.z_crop_slider.slider.setRange(
                (0, self.image.shape[self._metadata.axes.find("Z")])
            )
            self.z_crop_slider.slider.setValues(
                (0, self.image.shape[self._metadata.axes.find("Z")])
            )
            self.z_crop_slider.upper_limit_label.setText(
                str(self.image.shape[self._metadata.axes.find("Z")])
            )
        else:
            self.z_crop_slider.setHidden(True)

        if "T" in self._metadata.axes:
            self.t_crop_slider.setHidden(False)
            self.t_crop_slider.slider.setRange(
                (0, self.image.shape[self._metadata.axes.find("T")])
            )
            self.t_crop_slider.slider.setValues(
                (0, self.image.shape[self._metadata.axes.find("T")])
            )
            self.t_crop_slider.upper_limit_label.setText(
                str(self.image.shape[self._metadata.axes.find("T")])
            )
        else:
            self.t_crop_slider.setHidden(True)

    def initialize_viewer(self):
        self.viewer_model.layers.clear()
        self.viewer_model.add_image(self._image)

        self.crop_layer = self.viewer_model.add_image(self.selection_array)
        self.crop_layer.opacity = 0.4
        self.crop_layer.colormap = "cyan"

        for slider_widget in self.viewer_qt.dims.slider_widgets:
            slider_widget.layout().itemAt(0).widget().setMinimumWidth(75)
            slider_widget.layout().itemAt(1).widget().setMaximumWidth(75)
            slider_widget.layout().itemAt(1).widget().setText("Play")
            slider_widget.layout().itemAt(3).widget().setMinimumWidth(75)
            slider_widget.layout().itemAt(5).widget().setMinimumWidth(75)

        self.update_summary()

    def update_crop_label_layer(self):
        self.crop_layer.data = self.selection_array

    def update_summary(self):
        self.summary_nbvoxels_label.setText(
            f"Total number of voxels: {numpy.prod(self.cropped_image[0].shape)}"
        )
        self.summary_nbytes_label.setText(
            f"Total size in bytes: {human_readable_byte_size(self.cropped_image[0].nbytes)}"
        )

    def clear_cropping_tab(self):
        self.viewer_model.layers.clear()

        self.summary_nbvoxels_label.setText("Total number of voxels: N/A")
        self.summary_nbytes_label.setText("Total size in bytes: N/A")

    def on_data_model_update(self):
        self.images = self.parent.data_model.images_to_denoise
