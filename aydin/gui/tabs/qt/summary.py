from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.custom_widgets.system_summary import SystemSummaryWidget


class SummaryTab(QWidget):
    """
    Aydin -- Image denoising, but chill.

    Aydin is a user-friendly, feature-rich, and fast image denoising tool that provides
    a number of self-supervised, auto-tuned, and unsupervised image denoising algorithms.
    Aydin handles from the get-go n-dimensional array-structured images with an arbitrary number
    of batch dimensions, channel dimensions, and typically up to 4 spatio-temporal dimensions.
    <br><br>
    You can drag and drop an image anywhere on the window to start or click `Add File(s)`
    button on top left part of the window. You can also load any of the example images.
    Once one or several image files have been loaded, you can adjust your desired settings
    in each tab from left to right, detailed explanations are given for all tabs and settings.
    Click on the Start button to start denoising, click on the 'View Images' button to view images
    before and after denoising.

    <moreless>
    <br><br>
    To learn how to use Aydin Studio -- this user-friendly interface, check our
    <a href='https://royerlab.github.io/aydin/tutorials/tutorials_home.html'>tutorials</a>.
    To see examples of how to tune Aydin for a particular image, check our
    <a href='https://royerlab.github.io/aydin/use_cases/introduction.html'>use cases</a>.
    Finally, once you have trained an Aydin model using any of the algorithms provided, you can process a large
    number of files in a 'headless' fashion using our
    <a href='https://royerlab.github.io/aydin/tutorials/cli_tutorials.html'> command line interface</a>.
    """

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        self.explanation_layout = QHBoxLayout()

        # Explanation text
        self.explanation_text = QLabel(self.__doc__, self)
        self.explanation_text.setWordWrap(True)
        self.explanation_text.setTextFormat(Qt.RichText)
        self.explanation_text.setOpenExternalLinks(True)

        self.tabs_explanation_text = QLabel(
            """
            File(s) Tab        -> See list added file(s).\n
            Image(s) Tab       -> See corresponding image(s) and select which ones to denoise.\n
            Dimension Tab      -> Tell Aydin how to interpret the different image dimensions.\n
            Training Crop Tab  -> Crop your image(s) for the purpose of training the denoising model.\n
            Inference Crop Tab -> Crop your image(s) for the purpose of actual denoising.\n
            Denoise Tab        -> Select denoising algorithm and corresponding settings.\n
            """,
            self,
        )
        self.explanation_layout.addWidget(self.explanation_text)
        self.explanation_layout.addWidget(self.tabs_explanation_text)
        self.tab_layout.addLayout(self.explanation_layout)

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.tab_layout.addWidget(SystemSummaryWidget(self))

        self.tab_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.tab_layout)
