"""Structured logging utilities for Aydin.

Provides tree-structured logging with hierarchical sections, elapsed time
tracking, and optional GUI callback support. Uses the ``arbol`` library as
the backend for tree formatting.

Use ``aprint`` for log messages and ``asection`` as a context manager for
nested log sections. Legacy aliases ``lprint`` and ``lsection`` are also
available for backward compatibility.
"""

import math
import re
import sys
from contextlib import contextmanager

import click
from arbol import Arbol as _ArbolBase
from arbol import asection as _arbol_asection

# ANSI escape code pattern for stripping colors before GUI emission
_ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class _LogMeta(type):
    """Metaclass for Log that provides class-level property for depth."""

    @property
    def depth(cls):
        """Current nesting depth (read from arbol backend)."""
        return _ArbolBase._depth


class Log(metaclass=_LogMeta):
    """Singleton-style logging configuration class.

    Manages global logging state including output depth, GUI callbacks,
    and display settings. All attributes are class-level (static).

    This class wraps arbol.Arbol and adds Aydin-specific features:
    - GUI callback emission (Qt signals)
    - Test-aware output suppression
    - CLI click.echo support

    Attributes
    ----------
    gui_callback : object or None
        Qt signal for GUI log output.
    gui_statusbar : object or None
        Qt status bar widget for displaying messages.
    guiEnabled : bool
        Whether GUI logging is enabled.
    enable_output : bool
        Whether console output is enabled.
    depth : int
        Current nesting depth for tree-structured output (read-only property).
    max_depth : int or float
        Maximum depth to display (use ``math.inf`` for unlimited).
    log_elapsed_time : bool
        Whether to display elapsed time for sections.
    override_test_exclusion : bool
        When True, logging is enabled even during test runs.
    force_click_echo : bool
        When True, uses ``click.echo`` instead of ``print``.
    """

    # GUI integration
    gui_callback = None
    gui_statusbar = None
    guiEnabled = True

    # Output control
    enable_output = False
    override_test_exclusion = False
    force_click_echo = False

    # Timing
    log_elapsed_time = True

    # Max depth (synced to Arbol)
    max_depth = math.inf

    def __init__(self):
        return

    @staticmethod
    def native_print(*args, sep=' ', end='\n', file=None):
        """Print to stdout and optionally emit to GUI callback.

        This method handles three output channels:
        1. Console output (via print or click.echo)
        2. GUI callback emission (Qt signal)
        3. GUI status bar update

        ANSI escape codes are stripped before GUI emission since
        QTextEdit does not render them.

        Parameters
        ----------
        *args : object
            Values to print.
        sep : str
            Separator between values.
        end : str
            String appended after the last value.
        file : file-like
            Output stream for console printing (defaults to sys.stdout).
        """
        # Build the text string
        text = sep.join(str(arg) for arg in args) + end

        # Console output
        if Log.enable_output:
            if Log.force_click_echo:
                # click.echo adds its own newline, strip ours
                click.echo(text.rstrip('\n'))
            else:
                # Use sys.stdout if no file specified (allows pytest capture)
                output_file = file if file is not None else sys.stdout
                print(text, end='', file=output_file)

        # GUI callback emission
        if Log.guiEnabled and Log.gui_callback is not None:
            # Strip ANSI codes for GUI display
            clean_text = _ANSI_ESCAPE.sub('', text)
            Log.gui_callback.emit(clean_text)

            if Log.gui_statusbar is not None:
                Log.gui_statusbar.showMessage(clean_text.rstrip('\n'))

    @staticmethod
    @contextmanager
    def test_context():
        """Context manager that enables logging output during tests."""
        Log.override_test_exclusion = True
        Log.force_click_echo = True
        yield
        Log.override_test_exclusion = False
        Log.force_click_echo = False

    @staticmethod
    def set_log_elapsed_time(log_elapsed_time: bool):
        """Enable or disable elapsed time display for log sections.

        Parameters
        ----------
        log_elapsed_time : bool
            Whether to show elapsed time.
        """
        Log.log_elapsed_time = log_elapsed_time
        _ArbolBase.elapsed_time = log_elapsed_time

    @staticmethod
    def set_log_max_depth(max_depth: int):
        """Set the maximum nesting depth for log output.

        Parameters
        ----------
        max_depth : int
            Maximum depth level to display (1-based).
        """
        Log.max_depth = max(0, max_depth - 1)
        _ArbolBase.max_depth = Log.max_depth


def _is_test_run():
    """Check if we're running inside a test."""
    for arg in sys.argv:
        if 'test' in arg or 'pytest' in arg:
            return True
    return False


def _should_output():
    """Determine if output should be produced."""
    if Log.override_test_exclusion:
        return True
    return not _is_test_run()


def aprint(*args, sep=' ', end='\n'):
    """Print a log message at the current nesting depth.

    Output is suppressed during test runs unless ``Log.override_test_exclusion``
    is True. Messages are indented according to the current section depth.

    Parameters
    ----------
    *args : object
        Values to print, same semantics as built-in ``print``.
    sep : str
        Separator between values.
    end : str
        String appended after the last value.
    """
    if not _should_output():
        return

    if _ArbolBase._depth <= Log.max_depth:
        level = min(Log.max_depth, _ArbolBase._depth)
        # Build prefix with tree scaffold
        prefix = _ArbolBase._vl_ * int(level) + _ArbolBase._br_ + ' '
        text = sep.join(str(arg) for arg in args)
        Log.native_print(prefix + text, end=end)


@contextmanager
def asection(section_header: str):
    """Context manager for a named log section with timing.

    Creates a hierarchical log section that tracks elapsed time.
    Nested sections are indented in the tree-structured output.

    Parameters
    ----------
    section_header : str
        Header text displayed at the start of the section.

    Yields
    ------
    None
        Control is yielded to the caller's code block.
    """
    if not _should_output():
        yield
        return

    # Temporarily enable arbol output and sync settings
    old_enable = _ArbolBase.enable_output
    _ArbolBase.enable_output = True
    _ArbolBase.elapsed_time = Log.log_elapsed_time
    _ArbolBase.max_depth = Log.max_depth
    # Disable colors for cleaner output (ANSI stripping handles GUI anyway)
    _ArbolBase.colorful = False

    # Store original native_print and replace with ours
    original_native_print = _ArbolBase.native_print

    @staticmethod
    def aydin_native_print(text, *args, sep=' ', end='\n', file=None):
        """Aydin's native print that routes through Log.native_print."""
        if args:
            text = text + sep + sep.join(str(arg) for arg in args)
        Log.native_print(text, end=end)

    _ArbolBase.native_print = aydin_native_print

    try:
        with _arbol_asection(section_header):
            yield
    finally:
        # Restore original settings
        _ArbolBase.native_print = original_native_print
        _ArbolBase.enable_output = old_enable


# Legacy aliases for backward compatibility
lprint = aprint
lsection = asection
