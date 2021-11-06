# flake8: noqa

from aydin.gui.gui import run
import traceback
import warnings
import sys

if __name__ == "__main__":

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

    warnings.simplefilter("always")

    run('ver')
