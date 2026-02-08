"""Range slider widget with axis labels and limit displays."""

from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from aydin.gui._qt.custom_widgets.range_slider import QHRangeSlider


class QRangeSliderWithLabels(QWidget):
    """Horizontal range slider with axis label, limit displays, and a Select All button.

    Combines a ``QHRangeSlider`` with labels showing the current lower and
    upper bounds, a range label, and a button to reset to the full range.

    Parameters
    ----------
    parent : BaseCroppingTab
        The parent cropping tab widget.
    label : str, optional
        Axis label displayed next to the slider (e.g. 'X', 'Y', 'Z', 'T').
        Default is 'N/A'.
    size : int, optional
        Initial slider range maximum. Default is 100.
    min_length : int, optional
        Minimum allowed range length for X and Y axes to prevent
        degenerate crops. Default is 32.
    """

    def __init__(self, parent, label="N/A", size=100, min_length=32):
        super(QRangeSliderWithLabels, self).__init__(parent)
        self.parent = parent
        self.size = size
        self.min_length = min_length

        # Slider1
        self.widget_layout = QHBoxLayout()

        self.slider_label = QLabel(label, self)
        self.widget_layout.addWidget(self.slider_label, 0)
        self.two_columns = QLabel(":", self)
        self.widget_layout.addWidget(self.two_columns)

        self.slider = QHRangeSlider(
            initial_values=(0, size), data_range=(0, size), collapsible=False
        )

        self.slider.setStep(1.0)
        self.slider.rangeChanged.connect(self.slider_range_changed)
        self.slider.valuesChanged.connect(self.slider_value_changed)

        self.widget_layout.addWidget(self.slider, 80)

        self.lower_limit_label = QPushButton("0")
        self.lower_limit_label.setDisabled(True)
        self.widget_layout.addWidget(self.lower_limit_label, 5)

        self.upper_limit_label = QPushButton(str(size))
        self.upper_limit_label.setDisabled(True)
        self.widget_layout.addWidget(self.upper_limit_label, 5)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(
            lambda: self.slider.setValues(self.slider.range())
        )
        self.widget_layout.addWidget(self.select_all_button)

        self.range_label = QLabel(f"[0,{str(size)})")
        self.range_label.setDisabled(True)
        self.widget_layout.addWidget(self.range_label)

        self.setLayout(self.widget_layout)

    @property
    def lower_cutoff(self):
        """Current lower bound of the crop selection.

        Returns
        -------
        int
            Lower cutoff value.
        """
        return int(self.lower_limit_label.text())

    @property
    def upper_cutoff(self):
        """Current upper bound of the crop selection.

        Returns
        -------
        int
            Upper cutoff value.
        """
        return int(self.upper_limit_label.text())

    def slider_range_changed(self, event):
        """Update the range label when the slider range changes.

        Parameters
        ----------
        event : tuple
            New (min, max) range values.
        """
        self.range_label.setText(f"[0,{event[1]})")

    def slider_value_changed(self, event):
        """Handle slider value changes by updating labels and the viewer.

        Enforces a minimum range length for X and Y sliders and updates
        the crop overlay and summary in the parent cropping tab.

        Parameters
        ----------
        event : tuple
            New (min, max) slider values (unused directly; values are
            read from the slider).
        """
        lower, upper = self.slider.values()

        if self.slider_label.text() in ["X", "Y"] and upper - lower <= self.min_length:

            self.slider.setValues((self.lower_cutoff, self.upper_cutoff))
            return

        if self.slider_label.text() not in ["X", "Y"]:
            if str(int(lower)) != self.lower_limit_label.text():
                self.parent.update_current_viewer_dims(self.slider_label.text(), lower)

            if str(int(upper)) != self.upper_limit_label.text():
                self.parent.update_current_viewer_dims(self.slider_label.text(), upper)

        self.parent.update_crop_label_layer()
        self.lower_limit_label.setText(str(int(lower)))
        self.upper_limit_label.setText(str(int(upper)))
        self.parent.update_summary()
