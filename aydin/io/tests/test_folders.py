"""Tests for Aydin folder utility functions."""

from os import W_OK, access
from os.path import exists

from aydin.io.folders import get_cache_folder, get_home_folder, get_temp_folder


def test_home_folder():
    """Test that the home folder exists and is writable."""
    home_folder = get_home_folder()
    print(home_folder)
    assert exists(home_folder)
    assert access(home_folder, W_OK)


def test_temp_folder():
    """Test that the temp folder exists and is writable."""
    temp_folder = get_temp_folder()
    print(temp_folder)
    assert exists(temp_folder)
    assert access(temp_folder, W_OK)


def test_cache_folder():
    """Test that the cache folder exists and is writable."""
    cache_folder = get_cache_folder()
    print(cache_folder)
    assert exists(cache_folder)
    assert access(cache_folder, W_OK)
