import os
import tempfile
from os import makedirs
from os.path import join, exists
from sys import platform


def get_home_folder():
    from pathlib import Path

    home_folder = f"{Path.home()}"
    return home_folder


def get_temp_folder():

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
