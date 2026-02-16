"""Semi-transparent loading overlay widget with animated dots.

Adapted from https://wiki.python.org/moin/PyQt/A%20full%20widget%20waiting%20indicator
"""

from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import QWidget


class Overlay(QWidget):
    """Semi-transparent overlay widget with an animated loading indicator.

    Displays a dark overlay with animated dots to indicate that a
    long-running operation (e.g., denoising) is in progress.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    """

    def __init__(self, parent=None):
        """Initialize the overlay with a transparent background palette.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        # Use ColorRole.Window for Qt6 compatibility (was Background in Qt5)
        palette.setColor(QPalette.ColorRole.Window, Qt.transparent)
        self.setPalette(palette)
        self.timer = None
        self.counter = 0

    def paintEvent(self, event):
        """Draw the overlay background and animated loading dots.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(40, 45, 60, 197)))
        painter.setPen(QPen(Qt.PenStyle.NoPen))

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
        """Start the animation timer when the overlay becomes visible.

        Parameters
        ----------
        event : QShowEvent
            The show event.
        """
        self.timer = self.startTimer(100)
        self.counter = 0

    def timerEvent(self, event):
        """Advance the animation counter and trigger a repaint.

        Parameters
        ----------
        event : QTimerEvent
            The timer event.
        """
        self.counter += 1
        self.update()

    def hideEvent(self, event):
        """Stop the animation timer when the overlay is hidden.

        Parameters
        ----------
        event : QHideEvent
            The hide event.
        """
        if self.timer is not None:
            self.killTimer(self.timer)
            self.timer = None
