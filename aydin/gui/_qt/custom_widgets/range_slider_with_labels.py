from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton

from aydin.gui._qt.custom_widgets.range_slider import QHRangeSlider


class QRangeSliderWithLabels(QWidget):
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
        return int(self.lower_limit_label.text())

    @property
    def upper_cutoff(self):
        return int(self.upper_limit_label.text())

    def slider_range_changed(self, event):
        self.range_label.setText(f"[0,{event[1]})")

    def slider_value_changed(self, event):
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
