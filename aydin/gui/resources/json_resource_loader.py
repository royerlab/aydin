"""Utilities for loading JSON and file resources from the GUI resources directory."""

import json
import os

from aydin.util.log.log import aprint


def absPath(myPath):
    """Get the absolute path to a GUI resource file.

    Works both during development (relative to this module) and when
    packaged with PyInstaller (using the ``_MEIPASS`` temp folder).

    Parameters
    ----------
    myPath : str
        Relative path or filename of the resource.

    Returns
    -------
    str
        Absolute path to the resource file.
    """
    import sys

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        aprint("found MEIPASS: %s " % os.path.join(base_path, os.path.basename(myPath)))

        return os.path.join(base_path, os.path.basename(myPath))
    except Exception as e:
        aprint("did not find MEIPASS: %s " % e)

        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


class JSONResourceLoader:
    """Loader for JSON resource files in the GUI resources directory.

    Parameters
    ----------
    resource_file_name : str
        Name of the JSON file to load (e.g. 'tooltips.json').

    Attributes
    ----------
    json : dict
        The parsed JSON content.

    Raises
    ------
    ValueError
        If ``resource_file_name`` is None or empty.
    """

    def __init__(self, resource_file_name):
        if resource_file_name is None or resource_file_name == "":
            raise ValueError(
                "JSONResourceLoader has to be initiated with a resource file name argument."
            )

        with open(absPath(resource_file_name)) as json_file:
            self.json = json.load(json_file)
