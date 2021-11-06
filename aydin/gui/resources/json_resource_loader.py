import json
import os

from aydin.util.log.log import lprint


def absPath(myPath):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    import sys

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        lprint("found MEIPASS: %s " % os.path.join(base_path, os.path.basename(myPath)))

        return os.path.join(base_path, os.path.basename(myPath))
    except Exception as e:
        lprint("did not find MEIPASS: %s " % e)

        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


class JSONResourceLoader:
    def __init__(self, resource_file_name):
        if resource_file_name is None or resource_file_name == "":
            raise ValueError(
                "JSONResourceLoader has to be initiated with a resource file name argument."
            )

        with open(absPath(resource_file_name)) as json_file:
            self.json = json.load(json_file)
