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
    outputWritten(text)
        Emitted whenever text is written. ``text`` is the output string.
    """

    outputWritten = QtCore.Signal(str)

    def __init__(self, parent, stdout=True):
        """Initialize the output wrapper and redirect the specified stream.

        Parameters
        ----------
        parent : QObject
            The parent Qt object.
        stdout : bool, optional
            If True, wraps ``sys.stdout``. If False, wraps ``sys.stderr``.
            Default is True.
        """
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
        self.outputWritten.emit(text)

    def __getattr__(self, name):
        """Delegate attribute access to the original stream.

        Parameters
        ----------
        name : str
            Attribute name to look up on the wrapped stream.

        Returns
        -------
        object
            The attribute from the original stream.
        """
        return getattr(self._stream, name)

    def __enter__(self):
        """Enter the context manager (stream is already redirected)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and restore the original stream."""
        self._restore()
        return False

    def _restore(self):
        """Restore the original stdout or stderr stream."""
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass

    def __del__(self):
        """Restore the original stdout or stderr stream on deletion."""
        self._restore()
