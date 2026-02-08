"""JSON serialization utilities using jsonpickle.

Provides functions for encoding Python objects to indented JSON strings,
and saving/loading arbitrary Python objects as JSON files using jsonpickle.
"""

import json

import jsonpickle


def encode_indent(object):
    """Encode a Python object as a pretty-printed JSON string.

    Uses jsonpickle to serialize the object and then reformats
    with indentation and sorted keys for readability.

    Parameters
    ----------
    object : object
        Any jsonpickle-serializable Python object.

    Returns
    -------
    str
        Indented, sorted JSON string representation.
    """
    return json.dumps(json.loads(jsonpickle.encode(object)), indent=4, sort_keys=True)


def save_any_json(dict2save, path):
    """Save an arbitrary Python object to a JSON file using jsonpickle.

    Parameters
    ----------
    dict2save : object
        Python object to serialize and save.
    path : str or Path
        File path to write the JSON output.
    """
    frozen = jsonpickle.encode(dict2save)
    with open(path, 'w') as fp:
        json.dump(frozen, fp)


def load_any_json(path):
    """Load a Python object from a jsonpickle-encoded JSON file.

    Parameters
    ----------
    path : str or Path
        File path to read the JSON from.

    Returns
    -------
    object
        Deserialized Python object.
    """
    with open(path, 'r') as fp:
        read_dict = json.load(fp)

    return jsonpickle.decode(read_dict)
