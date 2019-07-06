import sys

from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QWidget, QPlainTextEdit


class OutLog:
    def __init__(self, edit, out=None, color=None):
        self.edit = edit
        self.out = out
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QTextCursor.End)
        self.edit.insertPlainText(m)

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)

    def flush(self):
        pass


class LogConsole(QPlainTextEdit):

    def __init__(self, parent):
        super(QPlainTextEdit, self).__init__()
        sys.stdout = OutLog(self, out=sys.stdout)
        self.setReadOnly(True)
