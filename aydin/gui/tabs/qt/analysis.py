from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)


class AnalysisTab(QWidget):
    """
    Analysis Tab
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        # Blind spots
        self.blind_spot_label = QLabel("blind spots: []")
        self.tab_layout.addWidget(self.blind_spot_label)

        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        # SNR estimate
        self.snr_estimate_label = QLabel("snr_estimate: ")
        self.tab_layout.addWidget(self.snr_estimate_label)

        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        # Resolution estimate
        self.resolution_estimate_label = QLabel("frequency resolution estimate: ")
        self.tab_layout.addWidget(self.resolution_estimate_label)

        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        # Noise Floor section
        self.noise_floor_label = QLabel("calculated noise floor: ")
        self.tab_layout.addWidget(self.noise_floor_label)

        self.noise_floor_viewers_layout = QHBoxLayout()

        self.viewer_model1 = ViewerModel()
        self.viewer_qt1 = QtViewer(self.viewer_model1, show_welcome_screen=False)
        self.noise_floor_viewers_layout.addWidget(self.viewer_qt1)

        self.viewer_model2 = ViewerModel()
        self.viewer_qt2 = QtViewer(self.viewer_model2, show_welcome_screen=False)
        self.noise_floor_viewers_layout.addWidget(self.viewer_qt2)

        self.tab_layout.addLayout(self.noise_floor_viewers_layout)
        
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)
