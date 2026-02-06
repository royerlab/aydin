"""Aydin logging utilities.

This module provides tree-structured logging with hierarchical sections
and timing information. Uses ``arbol`` as the backend.

Example
-------
>>> from aydin.util.log import Log, lprint, lsection
>>> Log.enable_output = True
>>> with lsection('Processing'):
...     lprint('Step 1 complete')
...     with lsection('Substep'):
...         lprint('Detail')
"""

from aydin.util.log.log import Log, aprint, asection, lprint, lsection

__all__ = ['Log', 'lprint', 'lsection', 'aprint', 'asection']
