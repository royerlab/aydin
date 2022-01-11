from PyQt5.QtWidgets import QMenu
from qtpy.QtWidgets import (
    QHBoxLayout,
    QWidget,
    QPushButton,
    QToolButton,
    QStyle,
    QGroupBox,
    QLabel,
)
from qtpy.QtCore import Qt

from aydin.io.datasets import examples_single


class QProgramFlowDiagramWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent
        self.highlightable_buttons = []

        self.layout = QHBoxLayout()

        # self.main_group_box = QGroupBox("Program Flow")
        self.layout = QHBoxLayout()
        self.layout.setSpacing(5)

        self.load_data_group_box = QGroupBox("Load data")
        self.load_data_group_box_layout = QHBoxLayout()
        self.load_data_group_box_layout.setSpacing(0)
        self.load_data_group_box_layout.setContentsMargins(3, -1, 3, -1)

        self.add_files_button = QPushButton("Add File(s)")
        self.load_data_group_box_layout.addWidget(self.add_files_button)

        self.load_data_group_box_layout.addWidget(QLabel("or"))

        self.load_sample_image_button = QPushButton("Load example")
        menu = QMenu()
        menu_items = {
            "Fountain": examples_single.fountain,
            "Mona Lisa": examples_single.monalisa,
            "Gauss": examples_single.gauss_noisy,
            "Periodic": examples_single.periodic_noise,
            "Chessboard": examples_single.brown_chessboard_gray,
            "HCR (Royer)": examples_single.royerlab_hcr,
            "Blastocyst Fracking (Maitre)": examples_single.maitre_mouse,
        }
        for item in menu_items.keys():
            action = menu.addAction(item)
            action.setIconVisibleInMenu(False)
        menu.triggered.connect(
            lambda x: self.parent.load_sample_image(menu_items[x.text()])
        )

        self.load_sample_image_button.setMenu(menu)
        self.load_data_group_box_layout.addWidget(self.load_sample_image_button)

        self.load_data_group_box.setLayout(self.load_data_group_box_layout)
        self.layout.addWidget(self.load_data_group_box)

        self.layout.addWidget(self.forward_button())

        # choose image options
        self.choose_image_options_group_box = QGroupBox("Choose image options")
        self.choose_image_options_group_box_layout = QHBoxLayout()
        self.choose_image_options_group_box_layout.setSpacing(5)
        self.choose_image_options_group_box_layout.setContentsMargins(3, -1, 3, -1)

        self.files_button = QPushButton("File(s)")
        self.highlightable_buttons.append(self.files_button)
        self.choose_image_options_group_box_layout.addWidget(self.files_button)

        self.images_button = QPushButton("Image(s)")
        self.highlightable_buttons.append(self.images_button)
        self.choose_image_options_group_box_layout.addWidget(self.images_button)

        self.dimensions_button = QPushButton("Dimensions")
        self.highlightable_buttons.append(self.dimensions_button)
        self.choose_image_options_group_box_layout.addWidget(self.dimensions_button)

        self.training_crop_button = QPushButton("Training Crop")
        self.highlightable_buttons.append(self.training_crop_button)
        self.choose_image_options_group_box_layout.addWidget(self.training_crop_button)

        self.inference_crop_button = QPushButton("Denoising Crop")
        self.highlightable_buttons.append(self.inference_crop_button)
        self.choose_image_options_group_box_layout.addWidget(self.inference_crop_button)

        self.choose_image_options_group_box.setLayout(
            self.choose_image_options_group_box_layout
        )

        self.layout.addWidget(self.choose_image_options_group_box)

        self.layout.addWidget(self.forward_button())

        self.processing_group_box = QGroupBox("Process")
        self.processing_group_box_layout = QHBoxLayout()
        self.processing_group_box_layout.setSpacing(5)
        self.processing_group_box_layout.setContentsMargins(3, -1, 3, -1)

        self.preprocess_button = QPushButton("Preprocessing")
        self.highlightable_buttons.append(self.preprocess_button)
        self.processing_group_box_layout.addWidget(self.preprocess_button)

        self.processing_group_box_layout.addWidget(self.forward_button())

        self.denoise_button = QPushButton("Denoise")
        self.highlightable_buttons.append(self.denoise_button)
        self.processing_group_box_layout.addWidget(self.denoise_button)

        self.processing_group_box_layout.addWidget(self.forward_button())

        self.postprocess_button = QPushButton("Postprocessing")
        self.highlightable_buttons.append(self.postprocess_button)
        self.processing_group_box_layout.addWidget(self.postprocess_button)

        self.processing_group_box.setLayout(self.processing_group_box_layout)

        self.layout.addWidget(self.processing_group_box)

        self.layout.setAlignment(Qt.AlignHCenter)

        self.setLayout(self.layout)

    def highlight_button(self, current_tab_name):
        self.reset_buttons()
        for button in self.highlightable_buttons:
            if current_tab_name.lower() in button.text().lower():
                button.setStyleSheet(
                    "background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #0897c7, stop:1 #586727)"
                )
        else:
            if "processing" in current_tab_name.lower():
                self.preprocess_button.setStyleSheet(
                    "background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #0897c7, stop:1 #586727)"
                )
                self.postprocess_button.setStyleSheet(
                    "background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #0897c7, stop:1 #586727)"
                )

    def reset_buttons(self):
        for button in self.highlightable_buttons:
            button.setStyleSheet("")

    @staticmethod
    def forward_button():
        button = QToolButton()
        button.setEnabled(False)
        button.setIcon(button.style().standardIcon(QStyle.SP_MediaSeekForward))

        return button
