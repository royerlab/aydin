import locale
import math
import sys
import time
from contextlib import contextmanager


class Log:
    """
    Custom Log class
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
        if Log.enable_output:
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

    def set_log_elapsed_time(log_elapsed_time: bool):
        Log.log_elapsed_time = log_elapsed_time

    def set_log_max_depth(max_depth: int):
        Log.max_depth = max(0, max_depth - 1)


def lprint(*args, sep=' ', end='\n'):
    """
    Log print

    Parameters
    ----------
    args : list
    sep : str
    end : str

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
    """
    Log section

    Parameters
    ----------
    section_header : str

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
