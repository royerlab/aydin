"""Tests for the JSONResourceLoader and abs_path utility."""

import os

import pytest

from aydin.gui.resources.json_resource_loader import JSONResourceLoader, abs_path


def test_abs_path_returns_existing_file():
    """abs_path should resolve to an existing file for known resources."""
    path = abs_path("tooltips.json")
    assert os.path.isfile(path)


def test_abs_path_falls_back_to_module_relative():
    """When _MEIPASS is absent, abs_path uses module-relative path."""
    import sys

    assert not hasattr(sys, '_MEIPASS')
    path = abs_path("tooltips.json")
    assert "gui/resources" in path


def test_json_resource_loader_loads_tooltips():
    """JSONResourceLoader should load and parse the tooltips JSON file."""
    loader = JSONResourceLoader("tooltips.json")
    assert isinstance(loader.json, dict)
    assert len(loader.json) > 0


def test_json_resource_loader_raises_on_none():
    """JSONResourceLoader should raise ValueError when given None."""
    with pytest.raises(ValueError, match="resource file name"):
        JSONResourceLoader(None)


def test_json_resource_loader_raises_on_empty_string():
    """JSONResourceLoader should raise ValueError when given empty string."""
    with pytest.raises(ValueError, match="resource file name"):
        JSONResourceLoader("")


def test_json_resource_loader_raises_on_missing_file():
    """JSONResourceLoader should raise FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        JSONResourceLoader("nonexistent_file_12345.json")
