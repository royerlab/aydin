"""Utilities for loading JSON and file resources from the GUI resources directory."""

import json
import os


def abs_path(myPath):
    """Get the absolute path to a GUI resource file.

    Parameters
    ----------
    myPath : str
        Relative path or filename of the resource.

    Returns
    -------
    str
        Absolute path to the resource file.
    """
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
        """Load and parse a JSON resource file.

        Parameters
        ----------
        resource_file_name : str
            Name of the JSON file to load (e.g. 'tooltips.json').

        Raises
        ------
        ValueError
            If ``resource_file_name`` is None or empty.
        """
        if resource_file_name is None or resource_file_name == "":
            raise ValueError(
                "JSONResourceLoader has to be initiated"
                " with a resource file name argument."
            )

        with open(abs_path(resource_file_name)) as json_file:
            self.json = json.load(json_file)
