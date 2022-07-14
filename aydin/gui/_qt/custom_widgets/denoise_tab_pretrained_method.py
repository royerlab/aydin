from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QCheckBox,
)

from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)


class DenoiseTabPretrainedMethodWidget(QWidget):
    def __init__(self, parent, loaded_it):
        super(DenoiseTabPretrainedMethodWidget, self).__init__(parent)

        self.parent = parent
        self.loaded_it = loaded_it
        self.name = loaded_it.__class__.__name__
        self.description = f"This is a pretrained model, namely uses the image translator: {loaded_it.__class__.__name__}, will not train anything new but will quickly infer on the images of your choice."

        # Widget layout
        self.main_layout = QHBoxLayout()
        self.tab_method_layout = QVBoxLayout()
        self.tab_method_layout.setAlignment(Qt.AlignTop)

        # Description Label
        self.description_scroll = QScrollArea()
        self.description_scroll.setStyleSheet("QScrollArea {border: none;}")
        self.description_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.description_scroll.setAlignment(Qt.AlignTop)
        self.description_label = QLabel(self.description)
        self.description_label.setWordWrap(True)

        self.description_label.setTextFormat(Qt.RichText)
        self.description_label.setOpenExternalLinks(True)

        self.description_label.setAlignment(Qt.AlignTop)
        self.description_scroll.setWidget(self.description_label)
        self.description_scroll.setWidgetResizable(True)
        self.description_scroll.setMinimumHeight(300)

        self.tab_method_layout.addWidget(self.description_scroll)

        self.right_side_vlayout = QVBoxLayout()
        self.right_side_vlayout.setAlignment(Qt.AlignTop)

        # Checkboxes
        self.save_json_and_model_layout = QHBoxLayout()
        self.save_json_and_model_layout.setAlignment(Qt.AlignLeft)

        self.save_json_checkbox = QCheckBox("Save denoising options (JSON)")
        self.save_json_checkbox.setChecked(True)
        self.save_json_and_model_layout.addWidget(self.save_json_checkbox)
        self.save_json_and_model_layout.addWidget(QVerticalLineBreakWidget(self))

        self.save_model_checkbox = QCheckBox("Save the trained model")
        self.save_model_checkbox.setChecked(True)
        self.save_json_and_model_layout.addWidget(self.save_model_checkbox)

        self.right_side_vlayout.addLayout(self.save_json_and_model_layout)

        self.main_layout.addLayout(self.tab_method_layout, 35)
        self.main_layout.addWidget(QVerticalLineBreakWidget(self))
        self.main_layout.addLayout(self.right_side_vlayout, 50)

        self.setLayout(self.main_layout)
