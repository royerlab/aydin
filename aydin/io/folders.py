"""Platform-specific folder path utilities.

This module provides functions to retrieve platform-specific paths for
home, temporary, and cache directories used by Aydin.
"""

import os
import tempfile
from os import makedirs
from os.path import exists, join
from sys import platform


def get_home_folder():
    """Return the current user's home directory path.

    Returns
    -------
    home_folder : str
        Absolute path to the user's home directory.
    """
    from pathlib import Path

    home_folder = f"{Path.home()}"
    return home_folder


def get_temp_folder():
    """Return the platform-specific temporary directory path.

    Creates the directory if it does not exist.

    Returns
    -------
    temp_folder : str or None
        Absolute path to the temporary directory, or None if unavailable.
    """

    temp_folder = None

    if platform == "linux" or platform == "linux2":
        temp_folder = tempfile.gettempdir()

    elif platform == "darwin":
        temp_folder = tempfile.gettempdir()

    elif platform == "win32":
        temp_folder = tempfile.gettempdir()

    try:
        makedirs(temp_folder)
    except Exception:
        pass

    if exists(temp_folder):
        return temp_folder
    else:
        return None


def get_cache_folder():
    """Return the platform-specific cache directory path.

    On Linux, uses ``~/.cache``. On macOS, uses ``~/Library/Caches``.
    On Windows, uses the ``LOCALAPPDATA`` environment variable.
    Creates the directory if it does not exist.

    Returns
    -------
    cache_folder : str or None
        Absolute path to the cache directory, or None if unavailable.
    """

    cache_folder = None

    if platform == "linux" or platform == "linux2":
        cache_folder = join(get_home_folder(), '.cache')

    elif platform == "darwin":
        cache_folder = join(get_home_folder(), '/Library/Caches')

    elif platform == "win32":
        cache_folder = join(get_home_folder(), os.getenv('LOCALAPPDATA'))

    try:
        makedirs(cache_folder)
    except Exception:
        pass

    if exists(cache_folder):
        return cache_folder
    else:
        return None
