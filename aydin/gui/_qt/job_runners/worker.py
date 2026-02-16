"""Worker thread and signal classes for executing background tasks."""

import sys
import traceback

from qtpy.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by a ``Worker`` thread during and after execution.

    Attributes
    ----------
    finished : Signal
        Emitted when the worker function completes (success or failure).
    error : Signal(tuple)
        Emitted on exception with ``(exctype, value, formatted_traceback)``.
    result : Signal(object)
        Emitted on success with the return value of the worker function.
    progress : Signal(str)
        Emitted by the worker function to report progress text.
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(str)


class Worker(QRunnable):
    """QRunnable wrapper for executing a function in a background thread.

    Automatically injects a ``progress_callback`` keyword argument into the
    function call and emits signals for result, error, and completion.

    Parameters
    ----------
    fn : callable
        The function to execute. Will receive ``progress_callback`` as a
        keyword argument.
    *args
        Positional arguments forwarded to ``fn``.
    **kwargs
        Keyword arguments forwarded to ``fn``.
    """

    def __init__(self, fn, *args, **kwargs):
        """Initialize the worker with the function to execute.

        Parameters
        ----------
        fn : callable
            The function to execute in a background thread.
        *args
            Positional arguments forwarded to ``fn``.
        **kwargs
            Keyword arguments forwarded to ``fn``.
        """
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """Execute the wrapped function and emit result or error signals."""

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
