"""Stream wrapper that redirects stdout/stderr to Qt signals."""

import sys

from qtpy import QtCore


class OutputWrapper(QtCore.QObject):
    """Wraps stdout or stderr to emit a Qt signal on each write.

    Redirects the specified stream so that all output is both forwarded
    to the original stream and emitted via the ``outputWritten`` signal,
    enabling the GUI activity log to capture console output.

    Parameters
    ----------
    parent : QObject
        The parent Qt object.
    stdout : bool, optional
        If True, wraps ``sys.stdout``. If False, wraps ``sys.stderr``.
        Default is True.

    Signals
    -------
    outputWritten(text, is_stdout)
        Emitted whenever text is written. ``text`` is the output string,
        ``is_stdout`` is True for stdout and False for stderr.
    """

    outputWritten = QtCore.Signal(object, object)

    def __init__(self, parent, stdout=True):
        QtCore.QObject.__init__(self, parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        """Write text to the original stream and emit the outputWritten signal.

        Parameters
        ----------
        text : str
            The text to write.
        """
        self._stream.write(text)
        self.outputWritten.emit(text, self._stdout)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass
