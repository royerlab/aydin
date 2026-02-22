"""Aydin logging utilities.

This module provides tree-structured logging with hierarchical sections
and timing information. Uses ``arbol`` as the backend.

Example
-------
>>> from aydin.util.log import Log, aprint, asection
>>> Log.enable_output = True
>>> with asection('Processing'):
...     aprint('Step 1 complete')
...     with asection('Substep'):
...         aprint('Detail')
"""

from aydin.util.log.log import Log, aprint, asection, lprint, lsection

__all__ = ['Log', 'aprint', 'asection', 'lprint', 'lsection']
