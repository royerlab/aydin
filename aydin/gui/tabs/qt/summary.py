"""Summary tab showing application overview and system information."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.custom_widgets.system_summary import SystemSummaryWidget
from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)

# Left column: what Aydin is and how to use it
_LEFT_COLUMN_HTML = (
    "<b>Aydin -- Image denoising, but chill.</b>"
    "<br><br>"
    "Aydin is a user-friendly, feature-rich, and fast image denoising tool "
    "that provides a number of self-supervised, auto-tuned, and unsupervised "
    "image denoising algorithms. Aydin handles from the get-go n-dimensional "
    "array-structured images with an arbitrary number of batch dimensions, "
    "channel dimensions, and typically up to 4 spatio-temporal dimensions."
    "<br><br>"
    "You can drag and drop an image anywhere on the window to start or click "
    "<code>Add File(s)</code> button on top left part of the window. You can "
    "also load any of the example images. Once one or several image files "
    "have been loaded, you can adjust your desired settings in each tab from "
    "left to right. Click on the Start button to start denoising, click on "
    "the 'View Images' button to view images before and after denoising."
)

# Right column: resources, citation, and support links
_RIGHT_COLUMN_HTML = (
    "<b>Resources</b>"
    "<br><br>"
    "To learn how to use Aydin Studio, check our "
    "<a href='https://royerlab.github.io/aydin/tutorials/tutorials_home.html'>"
    "tutorials</a>. "
    "To see examples of how to tune Aydin for a particular image, check our "
    "<a href='https://royerlab.github.io/aydin/use_cases/introduction.html'>"
    "use cases</a>. "
    "Finally, once you have trained an Aydin model, you can process a large "
    "number of files in a 'headless' fashion using our "
    "<a href='https://royerlab.github.io/aydin/tutorials/cli_tutorials.html'>"
    "command line interface</a>."
    "<br><br>"
    "If you find Aydin useful in your work, please kindly cite Aydin by using "
    "our DOI: "
    "<a href='https://doi.org/10.5281/zenodo.5654826'>10.5281/zenodo.5654826</a>."
    "<br><br>"
    "If you have any bug reports or feature requests, please reach us at "
    "<a href='https://github.com/royerlab/aydin'>our GitHub repository</a>. "
    "If you have any general question, please reach us at "
    "<a href='https://forum.image.sc/tag/aydin'>image.sc</a>."
)

# Width threshold below which columns stack vertically
_SINGLE_COLUMN_WIDTH = 900

# Maximum width for the text columns (keeps line lengths readable)
_MAX_COLUMNS_WIDTH = 1200


class SummaryTab(QWidget):
    """Summary tab with two-column description and system information."""

    def __init__(self, parent):
        """Initialize the Summary tab.

        Parameters
        ----------
        parent : MainPage
            The parent MainPage widget.
        """
        super().__init__(parent)
        self.parent = parent

        # Outer layout holds a scroll area so all content is accessible
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea {border: none;}")

        content = QWidget()
        self.tab_layout = QVBoxLayout()

        # Two-column description text
        self._two_column = True
        self.columns_widget = QWidget()
        self.columns_layout = QHBoxLayout()
        self.columns_layout.setContentsMargins(0, 0, 0, 0)
        self.columns_layout.setSpacing(20)

        self.left_text = self._make_label(_LEFT_COLUMN_HTML)
        self.column_separator = QVerticalLineBreakWidget(self)
        self.right_text = self._make_label(_RIGHT_COLUMN_HTML)

        self.columns_layout.addWidget(self.left_text, 1)
        self.columns_layout.addWidget(self.column_separator, 0)
        self.columns_layout.addWidget(self.right_text, 1)
        self.columns_widget.setLayout(self.columns_layout)
        self.columns_widget.setMaximumWidth(_MAX_COLUMNS_WIDTH)

        # Center the columns when the window is wider than the max width
        columns_center = QHBoxLayout()
        columns_center.setContentsMargins(0, 0, 0, 0)
        columns_center.addStretch(1)
        columns_center.addWidget(self.columns_widget, 1000)
        columns_center.addStretch(1)
        self.tab_layout.addLayout(columns_center)

        # Horizontal Line Break
        self.system_summary_separator = QHorizontalLineBreakWidget(self)
        self.tab_layout.addWidget(self.system_summary_separator)

        self.system_summary_widget = SystemSummaryWidget(self)
        self.tab_layout.addWidget(self.system_summary_widget)

        # Thresholds for auto-hiding system summary (computed on first resize)
        self._system_summary_min_height = 0
        self._system_summary_min_width = 0

        # Push content to the top without constraining widget heights.
        # (Qt.AlignTop would force sizeHint heights, clipping word-wrapped labels.)
        self.tab_layout.addStretch(1)
        content.setLayout(self.tab_layout)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, 1)

        self.setLayout(outer_layout)

    @staticmethod
    def _make_label(html):
        """Create a top-aligned, word-wrapping rich-text label."""
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setOpenExternalLinks(True)
        label.setAlignment(Qt.AlignTop)
        return label

    def _set_column_layout(self, two_column):
        """Switch between two-column (HBox) and single-column (VBox) layout."""
        if two_column == self._two_column:
            return
        self._two_column = two_column

        # Detach widgets from current layout
        self.columns_layout.removeWidget(self.left_text)
        self.columns_layout.removeWidget(self.column_separator)
        self.columns_layout.removeWidget(self.right_text)

        # Delete old layout by assigning to a temporary widget
        QWidget().setLayout(self.columns_widget.layout())

        if two_column:
            self.columns_layout = QHBoxLayout()
            self.columns_layout.setSpacing(20)
        else:
            self.columns_layout = QVBoxLayout()
            self.columns_layout.setSpacing(10)

        self.columns_layout.setContentsMargins(0, 0, 0, 0)
        if two_column:
            # Equal-width columns with separator
            self.columns_layout.addWidget(self.left_text, 1)
            self.columns_layout.addWidget(self.column_separator, 0)
            self.columns_layout.addWidget(self.right_text, 1)
        else:
            # Natural heights, no stretch
            self.columns_layout.addWidget(self.left_text)
            self.columns_layout.addWidget(self.column_separator)
            self.columns_layout.addWidget(self.right_text)
        self.column_separator.setVisible(two_column)
        self.columns_widget.setLayout(self.columns_layout)

    def resizeEvent(self, event):
        """Responsive layout: stack columns vertically when narrow,
        hide system summary when short.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        w = event.size().width()
        h = event.size().height()

        # Two columns when wide enough, single column when narrow
        self._set_column_layout(w >= _SINGLE_COLUMN_WIDTH)

        # Compute system summary thresholds on first resize
        if self._system_summary_min_height == 0:
            self._system_summary_min_height = (
                self.system_summary_widget.sizeHint().height()
                + self.left_text.sizeHint().height()
                + 100
            )
        if self._system_summary_min_width == 0:
            self._system_summary_min_width = (
                self.system_summary_widget.sizeHint().width() + 40
            )

        # System summary: hide when too short or too narrow
        show_summary = (
            h >= self._system_summary_min_height and w >= self._system_summary_min_width
        )
        if self.system_summary_widget.isVisible() != show_summary:
            self.system_summary_widget.setVisible(show_summary)
            self.system_summary_separator.setVisible(show_summary)

        event.accept()
