"""Tests for JSON serialization utilities."""

import json

import numpy as np

from aydin.util.misc.json import (
    _NumpyArrayHandler,
    _NumpyScalarHandler,
    encode_indent,
    load_any_json,
    save_any_json,
)

# --- NumpyScalarHandler roundtrip tests ---


def test_scalar_handler_float32():
    """Test NumpyScalarHandler roundtrip for float32 values."""
    handler = _NumpyScalarHandler(None)
    val = np.float32(3.14)
    data = handler.flatten(val, {})
    restored = handler.restore(data)
    assert isinstance(restored, np.float32)
    np.testing.assert_almost_equal(restored, val, decimal=5)


def test_scalar_handler_int64():
    """Test NumpyScalarHandler roundtrip for int64 values."""
    handler = _NumpyScalarHandler(None)
    val = np.int64(42)
    data = handler.flatten(val, {})
    restored = handler.restore(data)
    assert isinstance(restored, np.int64)
    assert restored == val


def test_scalar_handler_bool():
    """Test NumpyScalarHandler roundtrip for bool_ values."""
    handler = _NumpyScalarHandler(None)
    val = np.bool_(True)
    data = handler.flatten(val, {})
    restored = handler.restore(data)
    assert isinstance(restored, np.bool_)
    assert restored == val


def test_scalar_handler_complex128():
    """Test NumpyScalarHandler roundtrip for complex128 values."""
    handler = _NumpyScalarHandler(None)
    val = np.complex128(1.0 + 2.0j)
    data = handler.flatten(val, {})
    restored = handler.restore(data)
    assert isinstance(restored, np.complex128)
    assert restored == val


# --- NumpyArrayHandler roundtrip tests ---


def test_array_handler_1d_float64():
    """Test NumpyArrayHandler roundtrip for 1D float64 arrays."""
    handler = _NumpyArrayHandler(None)
    arr = np.array([1.0, 2.5, 3.7], dtype=np.float64)
    data = handler.flatten(arr, {})
    restored = handler.restore(data)
    np.testing.assert_array_equal(restored, arr)
    assert restored.dtype == arr.dtype


def test_array_handler_3d_uint8():
    """Test NumpyArrayHandler roundtrip for 3D uint8 arrays."""
    handler = _NumpyArrayHandler(None)
    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    data = handler.flatten(arr, {})
    restored = handler.restore(data)
    np.testing.assert_array_equal(restored, arr)
    assert restored.shape == (2, 3, 4)


def test_array_handler_empty():
    """Test NumpyArrayHandler roundtrip for empty float32 arrays."""
    handler = _NumpyArrayHandler(None)
    arr = np.array([], dtype=np.float32)
    data = handler.flatten(arr, {})
    restored = handler.restore(data)
    assert restored.shape == (0,)
    assert restored.dtype == np.float32


# --- encode_indent ---


def test_encode_indent_valid_json():
    """Test that encode_indent produces valid parseable JSON."""
    obj = {'key': 'value', 'number': 42}
    result = encode_indent(obj)
    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed['key'] == 'value'
    assert parsed['number'] == 42


def test_encode_indent_formatted():
    """Test that encode_indent output contains indentation newlines."""
    obj = {'a': 1}
    result = encode_indent(obj)
    # Should contain indentation
    assert '\n' in result


# --- save_any_json / load_any_json file roundtrip ---


def test_save_load_roundtrip(tmp_path):
    """Test save_any_json and load_any_json file roundtrip."""
    data = {'name': 'test', 'values': [1, 2, 3], 'nested': {'a': True}}
    path = str(tmp_path / 'test.json')
    save_any_json(data, path)
    restored = load_any_json(path)
    assert restored == data
