"""Tests for the DenoiseRestorationBase utility methods.

Tests the static methods for implementation discovery, argument extraction,
and model archiving/cleanup on the abstract base class.
"""

import pytest

from aydin.it import classic_denoisers
from aydin.restoration.denoise.base import DenoiseRestorationBase


def test_get_implementations_in_module():
    """Test discovery of submodules in a known package."""
    modules = DenoiseRestorationBase.get_implementations_in_a_module(classic_denoisers)
    assert isinstance(modules, list)
    assert len(modules) > 0
    # Check that known denoisers are found
    module_names = [m.name for m in modules]
    assert 'butterworth' in module_names
    assert 'gaussian' in module_names
    assert 'tv' in module_names
    # Should not include packages or 'base'
    for m in modules:
        assert not m.ispkg
        assert m.name != 'base'


def test_get_implementations_excludes_packages():
    """Test that get_implementations_in_a_module excludes sub-packages."""
    modules = DenoiseRestorationBase.get_implementations_in_a_module(classic_denoisers)
    for m in modules:
        assert not m.ispkg


def test_get_function_implementation_kwargs():
    """Test kwargs extraction from a known calibration function."""
    modules = DenoiseRestorationBase.get_implementations_in_a_module(classic_denoisers)
    # Find the butterworth module
    butterworth_mod = [m for m in modules if m.name == 'butterworth'][0]

    result = DenoiseRestorationBase.get_function_implementation_kwargs(
        classic_denoisers, butterworth_mod, 'calibrate_denoise_butterworth'
    )
    assert 'arguments' in result
    assert 'defaults' in result
    assert 'annotations' in result
    assert 'reference_class' in result
    assert isinstance(result['arguments'], list)
    assert len(result['arguments']) > 0


def test_get_class_implementation_kwargs():
    """Test kwargs extraction from a known class."""
    from aydin import regression

    modules = DenoiseRestorationBase.get_implementations_in_a_module(regression)
    # Find cb module
    cb_mod = [m for m in modules if m.name == 'cb'][0]

    result = DenoiseRestorationBase.get_class_implementation_kwargs(
        regression, cb_mod, 'CBRegressor'
    )
    assert 'arguments' in result
    assert 'defaults' in result
    assert 'reference_class' in result
    assert isinstance(result['arguments'], list)


def test_get_class_implementation_kwargs_not_found():
    """Test ValueError when no matching class is found."""
    from aydin import regression

    modules = DenoiseRestorationBase.get_implementations_in_a_module(regression)
    cb_mod = [m for m in modules if m.name == 'cb'][0]

    with pytest.raises(ValueError, match="No class matching"):
        DenoiseRestorationBase.get_class_implementation_kwargs(
            regression, cb_mod, 'NonExistentClassName'
        )


def test_clean_model_folder(tmp_path):
    """Test that clean_model_folder removes the folder and contents."""
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()
    (model_dir / 'weights.bin').write_bytes(b'fake_weights')
    (model_dir / 'config.json').write_text('{}')

    assert model_dir.exists()
    DenoiseRestorationBase.clean_model_folder(str(model_dir))
    assert not model_dir.exists()


def test_archive_and_extract(tmp_path):
    """Test zip archiving round-trip."""
    # Create source directory with files
    source_dir = tmp_path / 'source_model'
    source_dir.mkdir()
    (source_dir / 'data.txt').write_text('hello')
    (source_dir / 'weights.bin').write_bytes(b'\x00\x01\x02')

    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()

    # Archive
    DenoiseRestorationBase.archive(str(source_dir), str(dest_dir))

    # Verify zip was created
    archive_path = dest_dir / 'source_model.zip'
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0


def test_archive_overwrites_existing(tmp_path):
    """Test that archiving overwrites an existing archive."""
    source_dir = tmp_path / 'model'
    source_dir.mkdir()
    (source_dir / 'data.txt').write_text('version1')

    dest_dir = tmp_path / 'dest'
    dest_dir.mkdir()

    # Create first archive
    DenoiseRestorationBase.archive(str(source_dir), str(dest_dir))
    archive_path = dest_dir / 'model.zip'
    assert archive_path.exists()
    # Update source and re-archive
    (source_dir / 'extra.txt').write_text('extra data for version2')
    DenoiseRestorationBase.archive(str(source_dir), str(dest_dir))
    assert archive_path.exists()
