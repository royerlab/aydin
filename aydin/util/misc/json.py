"""JSON serialization utilities using jsonpickle.

Provides functions for encoding Python objects to indented JSON strings,
and saving/loading arbitrary Python objects as JSON files using jsonpickle.

Registers custom jsonpickle handlers for numpy types to work around
compatibility issues between jsonpickle 3.x and numpy 2.x where
scalar and ndarray ``__reduce__``/``__setstate__`` protocols changed.
"""

import base64
import json

import jsonpickle
import numpy as np


class _NumpyScalarHandler(jsonpickle.handlers.BaseHandler):
    """Serialize numpy scalars as Python native types for jsonpickle.

    numpy 2.x changed the internal ``scalar()`` constructor signature,
    breaking jsonpickle's default ``__reduce__``-based serialization.
    This handler stores value + dtype as plain JSON.
    """

    def flatten(self, obj, data):
        """Serialize a numpy scalar to a JSON-compatible dictionary.

        Parameters
        ----------
        obj : numpy.generic
            Numpy scalar value to serialize.
        data : dict
            jsonpickle state dictionary to populate.

        Returns
        -------
        dict
            Updated state dictionary with ``'value'`` and ``'dtype'`` keys.
        """
        data['value'] = obj.item()
        data['dtype'] = str(obj.dtype)
        return data

    def restore(self, obj):
        """Deserialize a numpy scalar from a JSON-compatible dictionary.

        Parameters
        ----------
        obj : dict
            Dictionary with ``'value'`` and ``'dtype'`` keys.

        Returns
        -------
        numpy.generic
            Restored numpy scalar.
        """
        return np.dtype(obj['dtype']).type(obj['value'])


class _NumpyArrayHandler(jsonpickle.handlers.BaseHandler):
    """Serialize numpy ndarrays for jsonpickle with numpy 2.x compatibility.

    numpy 2.x changed ``ndarray.__setstate__`` to expect 4 elements instead
    of 5, breaking jsonpickle's default ``__reduce__``-based roundtrip.
    This handler stores arrays as base64-encoded bytes with dtype and shape.
    """

    def flatten(self, obj, data):
        """Serialize a numpy ndarray to a JSON-compatible dictionary.

        Stores the array as base64-encoded raw bytes alongside its
        dtype and shape metadata.

        Parameters
        ----------
        obj : numpy.ndarray
            Array to serialize.
        data : dict
            jsonpickle state dictionary to populate.

        Returns
        -------
        dict
            Updated state dictionary with ``'dtype'``, ``'shape'``, and
            ``'data'`` keys.
        """
        data['dtype'] = str(obj.dtype)
        data['shape'] = list(obj.shape)
        data['data'] = base64.b64encode(obj.tobytes()).decode('ascii')
        return data

    def restore(self, obj):
        """Deserialize a numpy ndarray from a JSON-compatible dictionary.

        Decodes the base64 data and reconstructs the array with the
        original dtype and shape.

        Parameters
        ----------
        obj : dict
            Dictionary with ``'dtype'``, ``'shape'``, and ``'data'`` keys.

        Returns
        -------
        numpy.ndarray
            Restored numpy array.
        """
        dtype = np.dtype(obj['dtype'])
        shape = tuple(obj['shape'])
        buf = base64.b64decode(obj['data'])
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


# Register handlers for numpy scalars
for _scalar_type in (
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
    np.complex64,
    np.complex128,
):
    jsonpickle.handlers.register(_scalar_type, _NumpyScalarHandler)

# Register handler for numpy ndarrays
jsonpickle.handlers.register(np.ndarray, _NumpyArrayHandler)


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
