from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel

from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)


class QReadMoreLessLabel(QWidget):
    def __init__(self, parent, text):
        QWidget.__init__(self, parent)

        self.readmore = False

        # Explanation text
        self.explanation_layout = QHBoxLayout()
        self.explanation_layout.setAlignment(Qt.AlignTop)
        self.mousePressEvent = self.state_toggle

        self.readless_text, self.readmore_text = text.split("<moreless>")
        self.readmore_text = text

        self.readmore_left, self.readmore_right = self.readmore_text.split("<split>")

        self.explanation_text_left = QLabel(self.readless_text, self)
        self.explanation_text_left.setWordWrap(True)
        self.explanation_text_left.setTextFormat(Qt.RichText)
        self.explanation_layout.addWidget(self.explanation_text_left, 50)

        # Vertical Line Break
        self.vertical_break = QVerticalLineBreakWidget(self)
        self.explanation_layout.addWidget(self.vertical_break)

        self.explanation_text_right = QLabel("", self)
        self.explanation_text_right.setWordWrap(True)
        self.explanation_text_right.setTextFormat(Qt.RichText)
        self.explanation_layout.addWidget(self.explanation_text_right, 50)

        if self.readmore_right.strip() == "":
            self.vertical_break.setHidden(True)
            self.explanation_text_right.setHidden(True)

        self.setLayout(self.explanation_layout)

    def state_toggle(self, a0: QMouseEvent) -> None:
        self.readmore = not self.readmore
        if self.readmore:
            self.explanation_text_left.setText(self.readmore_left)
            self.explanation_text_right.setText(self.readmore_right)
        else:
            self.explanation_text_left.setText(self.readless_text)
            self.explanation_text_right.setText("")
