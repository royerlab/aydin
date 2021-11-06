"""
Adapted from https://wiki.python.org/moin/PyQt/A%20full%20widget%20waiting%20indicator
"""
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette, QPainter, QBrush, QColor, QPen
from qtpy.QtWidgets import QWidget


class Overlay(QWidget):
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(40, 45, 60, 197)))
        painter.setPen(QPen(Qt.NoPen))

        for i in range(5):
            if (self.counter / 5) % 5 >= i:
                painter.setBrush(QBrush(QColor(0, 191, 255)))
            else:
                painter.setBrush(QBrush(QColor(197, 197, 197)))
            painter.drawEllipse(
                self.width() // 2 + 50 * i - 100, self.height() // 2, 20, 20
            )

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(100)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()

    def hideEvent(self, event):
        self.killTimer(self.timer)
        self.hide()
