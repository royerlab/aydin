"""Structured logging utilities for Aydin.

Provides tree-structured logging with hierarchical sections, elapsed time
tracking, and optional GUI callback support. Use ``lprint`` for log messages
and ``lsection`` as a context manager for nested log sections.
"""

import locale
import math
import sys
import time
from contextlib import contextmanager

import click


class Log:
    """Singleton-style logging configuration class.

    Manages global logging state including output depth, GUI callbacks,
    and display settings. All attributes are class-level (static).

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
        Current nesting depth for tree-structured output.
    max_depth : int or float
        Maximum depth to display (use ``math.inf`` for unlimited).
    log_elapsed_time : bool
        Whether to display elapsed time for sections.
    override_test_exclusion : bool
        When True, logging is enabled even during test runs.
    force_click_echo : bool
        When True, uses ``click.echo`` instead of ``print``.
    """

    # current_section = ''
    gui_callback = None
    gui_statusbar = None
    guiEnabled = True
    enable_output = False
    depth = 0
    max_depth = math.inf
    log_elapsed_time = True
    override_test_exclusion = False
    force_click_echo = False

    # Define special characters:
    __vl__ = '│'  # 'Vertical Line'
    __br__ = '├'  # 'Branch Right'
    __bd__ = '├╗'  # 'Branch Down'
    __tb__ = '┴'  # 'Terminate Branch'
    __la__ = '«'  # 'Left Arrow'

    #  Windows terminal is dumb. We can't use our fancy characters from Yesteryears, sad:
    if (
        locale.getpreferredencoding() == "US-ASCII"
        or locale.getpreferredencoding() == "cp1252"
    ):
        __vl__ = '|'
        __br__ = '|->'
        __bd__ = '|\ '  # noqa: W605
        __tb__ = '-'
        __la__ = '<<'

    def __init__(self):
        return

    @staticmethod
    def native_print(*args, sep=' ', end='\n', file=sys.__stdout__):
        """Print to stdout and optionally emit to GUI callback.

        Parameters
        ----------
        *args : object
            Values to print.
        sep : str
            Separator between values.
        end : str
            String appended after the last value.
        file : file-like
            Output stream for console printing.
        """
        if Log.enable_output:
            if Log.force_click_echo:
                click.echo(*args)
            else:
                print(*args, sep=sep, end=end, file=file)

        if Log.guiEnabled and Log.gui_callback is not None:
            result = ""
            for arg in args:
                result += str(arg)
                result += sep

            result += end
            Log.gui_callback.emit(result)

            if Log.gui_statusbar is not None:
                Log.gui_statusbar.showMessage(result)

    @staticmethod
    @contextmanager
    def test_context():
        """Context manager that enables logging output during tests."""
        Log.override_test_exclusion = True
        Log.force_click_echo = True
        yield
        Log.override_test_exclusion = False
        Log.force_click_echo = False

    def set_log_elapsed_time(log_elapsed_time: bool):
        """Enable or disable elapsed time display for log sections.

        Parameters
        ----------
        log_elapsed_time : bool
            Whether to show elapsed time.
        """
        Log.log_elapsed_time = log_elapsed_time

    def set_log_max_depth(max_depth: int):
        """Set the maximum nesting depth for log output.

        Parameters
        ----------
        max_depth : int
            Maximum depth level to display (1-based).
        """
        Log.max_depth = max(0, max_depth - 1)


def lprint(*args, sep=' ', end='\n'):
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
    if not Log.override_test_exclusion:
        for arg in sys.argv:
            if "test" in arg:
                return

    if Log.depth <= Log.max_depth:
        level = min(Log.max_depth, Log.depth)
        Log.native_print(Log.__vl__ * int(level) + Log.__br__ + ' ', end='')
        Log.native_print(*args, sep=sep, end=end)


@contextmanager
def lsection(section_header: str):
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
    if not Log.override_test_exclusion:
        for arg in sys.argv:
            if "test" in arg:
                yield
                return

    if Log.depth + 1 <= Log.max_depth:
        Log.native_print(
            Log.__vl__ * Log.depth + Log.__bd__ + ' ' + section_header
        )  # ≡
    elif Log.depth + 1 == Log.max_depth + 1:
        Log.native_print(
            Log.__vl__ * Log.depth
            + Log.__br__
            + f'= {section_header} (log tree truncated here)'
        )

    Log.depth += 1

    start = time.time()
    exception = None
    try:
        yield
    except Exception as e:
        exception = e

    stop = time.time()

    Log.depth -= 1
    if Log.depth + 1 <= Log.max_depth:

        if Log.log_elapsed_time:
            elapsed = stop - start

            if elapsed < 0.001:
                Log.native_print(
                    Log.__vl__ * (Log.depth + 1)
                    + Log.__tb__
                    + Log.__la__
                    + f' {elapsed * 1000 * 1000:.2f} microseconds'
                )
            elif elapsed < 1:
                Log.native_print(
                    Log.__vl__ * (Log.depth + 1)
                    + Log.__tb__
                    + Log.__la__
                    + f' {elapsed * 1000:.2f} milliseconds'
                )
            elif elapsed < 60:
                Log.native_print(
                    Log.__vl__ * (Log.depth + 1)
                    + Log.__tb__
                    + Log.__la__
                    + f' {elapsed:.2f} seconds'
                )
            elif elapsed < 60 * 60:
                Log.native_print(
                    Log.__vl__ * (Log.depth + 1)
                    + Log.__tb__
                    + Log.__la__
                    + f' {elapsed / 60:.2f} minutes'
                )
            elif elapsed < 24 * 60 * 60:
                Log.native_print(
                    Log.__vl__ * (Log.depth + 1)
                    + Log.__tb__
                    + Log.__la__
                    + f' {elapsed / (60 * 60):.2f} hours'
                )

        Log.native_print(Log.__vl__ * (Log.depth + 1))

        if exception is not None:
            raise exception
