from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel

from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)


class QReadMoreLessLabel(QWidget):
    def __init__(self, parent, text):
        QWidget.__init__(self, parent)

        self.text = text
        self.readmore = False

        # Explanation text
        self.explanation_layout = QHBoxLayout()
        self.explanation_layout.setAlignment(Qt.AlignTop)
        self.mousePressEvent = self.state_toggle

        if "<moreless>" in text:
            self.readless_text, self.readmore_text = text.split("<moreless>")
            self.readmore_text = text
            self.readmore_left, self.readmore_right = self.readmore_text.split(
                "<split>"
            )
            text_left = self.readless_text + (
                "" if self.readmore_right.strip() == "" else "<b>Read more...</b>"
            )
            text_right = ""
        else:
            text_left, text_right = text.split("<split>")

        self.explanation_text_left = QLabel(text_left, self)
        self.explanation_text_left.setWordWrap(True)
        self.explanation_text_left.setTextFormat(Qt.RichText)
        self.explanation_layout.addWidget(self.explanation_text_left, 50)

        # Vertical Line Break
        self.vertical_break = QVerticalLineBreakWidget(self)
        self.explanation_layout.addWidget(self.vertical_break)

        self.explanation_text_right = QLabel(text_right, self)
        self.explanation_text_right.setWordWrap(True)
        self.explanation_text_right.setTextFormat(Qt.RichText)
        self.explanation_layout.addWidget(self.explanation_text_right, 50)

        if self.readmore_text is not None and self.readmore_text.strip() == "":
            self.vertical_break.setHidden(True)
            self.explanation_text_right.setHidden(True)

        self.setLayout(self.explanation_layout)

    def state_toggle(self, a0: QMouseEvent) -> None:

        if self.readmore_text is not None and self.readmore_right.strip() != "":
            self.readmore = not self.readmore
            if self.readmore:
                self.explanation_text_left.setText(self.readmore_left)
                self.explanation_text_right.setText(
                    self.readmore_right + "<b>Read less...</b>"
                )
            else:
                self.explanation_text_left.setText(
                    self.readless_text + "<b>Read more...</b>"
                )
                self.explanation_text_right.setText("")
