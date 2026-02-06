# Shared hook utilities for PyInstaller builds
# This module provides common functions used by platform-specific hooks

import os
import re

from PyInstaller.compat import is_darwin, is_linux, is_win
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    exec_statement,
)


def _module_path(mod):
    """Get the installation path of a module.

    Parameters
    ----------
    mod : str
        Module name to get the path for.

    Returns
    -------
    str
        Directory path where the module is installed.
    """
    return exec_statement(
        """
        import sys
        import os
        _tmp = sys.stdout
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        import %s
        sys.stdout = _tmp
        print(os.path.dirname(%s.__file__))
        """
        % (mod, mod)
    )


def _get_toc_objects(
    root,
    filter_str=".*",
    dir_prefix="",
    flatten_dir=False,
):
    """Collect TOC (Table of Contents) objects from a directory.

    Parameters
    ----------
    root : str
        Root directory to walk.
    filter_str : str
        Regex pattern to filter filenames.
    dir_prefix : str
        Prefix to add to destination paths.
    flatten_dir : bool
        If True, flatten all files to the same directory level.

    Returns
    -------
    list
        List of (source_path, dest_path) tuples.
    """
    reg = re.compile(filter_str)
    res = []
    for fold, subs, files in os.walk(root):
        rel_dir = os.path.relpath(fold, root)
        for fName in files:
            if reg.match(fName):
                if not flatten_dir:
                    name = os.path.join(dir_prefix, rel_dir, fName)
                else:
                    name = os.path.join(dir_prefix, fName)
                res.append((os.path.join(fold, fName), name))
    return res


def _my_collect_data_files(modname, flatten_dirs=False, **kwargs):
    """Collect data files for a module with optional directory flattening.

    Parameters
    ----------
    modname : str
        Module name to collect data files from.
    flatten_dirs : bool
        If True, flatten all files to the root destination.
    **kwargs
        Additional arguments passed to collect_data_files.

    Returns
    -------
    list
        List of (source_path, dest_path) tuples.
    """
    files = collect_data_files(modname, **kwargs)
    if flatten_dirs:
        files = [(source, ".") for source, dest in files]
    return files


def get_lightgbm_binary():
    """Get the correct lightgbm binary name for the current platform.

    Returns
    -------
    str
        Binary filename (lib_lightgbm.dll, lib_lightgbm.dylib, or lib_lightgbm.so).
    """
    if is_win:
        return "lib_lightgbm.dll"
    elif is_darwin:
        return "lib_lightgbm.dylib"
    else:
        return "lib_lightgbm.so"


def get_platform_binary_extension():
    """Get the platform-specific extension for compiled Python modules.

    Returns
    -------
    str
        Extension pattern (e.g., '*.pyd' for Windows, '*.so' for Unix).
    """
    if is_win:
        return "*.pyd"
    else:
        return "*.so"


if __name__ == '__main__':
    # Test functions
    print("Platform:", "win" if is_win else "darwin" if is_darwin else "linux")
    print("LightGBM binary:", get_lightgbm_binary())
    print("Binary extension:", get_platform_binary_extension())
