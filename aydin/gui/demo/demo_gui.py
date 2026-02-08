"""Demo script for launching Aydin Studio GUI."""

# flake8: noqa

import sys
import traceback
import warnings

from aydin.gui.gui import run

if __name__ == "__main__":

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that prints the full stack trace."""
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

    warnings.simplefilter("always")

    run('ver')
