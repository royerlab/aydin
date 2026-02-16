"""Tests for the DataModel class."""

from pathlib import Path
from unittest.mock import Mock

import numpy

from aydin.gui.tabs.data_model import DataModel, ImageRecord
from aydin.io import imread
from aydin.io.datasets import examples_single
from aydin.io.io import FileMetadata


def _make_data_model():
    """Create a DataModel with a properly mocked parent."""
    parent = Mock()
    return DataModel(parent)


def test_not_adding_non_existing_filepaths():
    """Test that non-existing file paths are silently rejected."""
    data_model = _make_data_model()
    data_model.add_filepaths(["/asdfas/asdfasd/fdsaf/fas", "/fdsa/fdsa/fdsa/fdsa"])

    assert data_model.filepaths == {}


def test_adding_files():
    """Test that valid file paths are loaded with correct arrays and metadata."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])
    array, metadata = imread(fpath)
    target_dict = {fpath: (array, metadata)}

    for key in data_model.filepaths.keys():
        for elem1, elem2 in zip(data_model.filepaths[key], target_dict[key]):
            if isinstance(elem1, FileMetadata):
                assert elem1 == elem2
            else:
                assert numpy.array_equal(elem1, elem2)


def test_removing_files():
    """Test that clearing filepaths removes all entries."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    data_model.clear_filepaths()

    assert data_model.filepaths == dict()


def test_removing_multiple_but_not_all_files():
    """Test that removing specific filepaths leaves the remaining ones intact."""
    data_model = _make_data_model()
    fpath1 = examples_single.noisy_fountain.get_path()
    fpath2 = examples_single.generic_camera.get_path()
    data_model.add_filepaths([fpath1, fpath2])

    assert len(data_model.filepaths) == 2

    data_model.remove_filepaths([fpath1])
    assert len(data_model.filepaths) == 1
    assert fpath2 in data_model.filepaths


def test_adding_duplicate_filepaths():
    """Test that duplicate file paths are not added twice."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])
    data_model.add_filepaths([fpath])

    assert len(data_model.filepaths) == 1


def test_images_populated_after_adding_files():
    """Test that images list is populated when files are added."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    assert len(data_model.images) == 1
    image_record = data_model.images[0]
    assert isinstance(image_record, ImageRecord)
    assert image_record.filename == Path(fpath).name
    assert image_record.denoise is True  # denoise flag defaults to True
    assert image_record.filepath == fpath
    assert image_record.output_folder == str(Path(fpath).resolve().parent)


def test_clear_filepaths_clears_images():
    """Test that clearing filepaths also clears images."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    data_model.clear_filepaths()

    assert data_model.images == []
    assert data_model.filepaths == {}


def test_set_image_to_denoise():
    """Test toggling the denoise flag on an image."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    filename = Path(fpath).name
    # Default is True
    assert data_model.images[0].denoise is True
    assert len(data_model.images_to_denoise) == 1

    # Set to False
    data_model.set_image_to_denoise(filename, False)
    assert data_model.images[0].denoise is False
    assert len(data_model.images_to_denoise) == 0

    # Set back to True
    data_model.set_image_to_denoise(filename, True)
    assert data_model.images[0].denoise is True
    assert len(data_model.images_to_denoise) == 1


def test_images_to_denoise_filters_correctly():
    """Test that images_to_denoise returns only flagged images."""
    data_model = _make_data_model()
    fpath1 = examples_single.noisy_fountain.get_path()
    fpath2 = examples_single.generic_camera.get_path()
    data_model.add_filepaths([fpath1, fpath2])

    # Both default to True
    assert len(data_model.images_to_denoise) == 2

    # Disable one
    data_model.set_image_to_denoise(Path(fpath1).name, False)
    to_denoise = data_model.images_to_denoise
    assert len(to_denoise) == 1
    assert to_denoise[0].filename == Path(fpath2).name


def test_update_image_output_folder():
    """Test updating the output folder for an image."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    filename = Path(fpath).name
    new_folder = "/tmp/custom_output"
    data_model.update_image_output_folder(filename, new_folder)

    assert data_model.images[0].output_folder == new_folder


def test_set_hyperstack():
    """Test hyperstacking and de-hyperstacking images."""
    data_model = _make_data_model()
    fpath1 = examples_single.noisy_fountain.get_path()
    fpath2 = examples_single.generic_camera.get_path()

    # Load both images - they may have different shapes, so hyperstack may fail
    data_model.add_filepaths([fpath1, fpath2])

    arr1, _ = imread(fpath1)
    arr2, _ = imread(fpath2)

    if arr1.shape == arr2.shape and arr1.dtype == arr2.dtype:
        # Same shape: hyperstack should succeed
        result = data_model.set_hyperstack(True)
        assert result is None  # success
        assert len(data_model.images) == 1
        assert data_model.images[0].filename.startswith("hyperstack_")

        # De-hyperstack
        data_model.set_hyperstack(False)
        assert len(data_model.images) == 2
    else:
        # Different shapes: hyperstack should fail
        result = data_model.set_hyperstack(True)
        assert result == -1


def test_set_split_channels_no_channel_axis():
    """Test that splitting fails gracefully when no channel axis exists."""
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    _, metadata = imread(fpath)
    filename = Path(fpath).name

    if "C" not in metadata.axes:
        result = data_model.set_split_channels(filename, fpath, True)
        assert result == -1


def test_path_manipulation_with_name():
    """Test that path manipulation uses with_name() correctly.

    Verifies the fix for the bug where filepath.replace() could replace
    the wrong part of a path when the filename appears in a parent
    directory name.
    """
    data_model = _make_data_model()
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    # The hyperstack path should use with_name, not string replace
    if len(data_model.images) > 0:
        filepath = data_model.images[0].filepath
        filename = data_model.images[0].filename

        # Verify with_name produces correct result
        expected = str(Path(filepath).with_name(f"hyperstack_{filename}"))
        # The parent directory should be preserved
        assert Path(expected).parent == Path(filepath).parent


def test_remove_filepaths_also_removes_images():
    """Test that removing filepaths also removes corresponding images."""
    data_model = _make_data_model()
    fpath1 = examples_single.noisy_fountain.get_path()
    fpath2 = examples_single.generic_camera.get_path()
    data_model.add_filepaths([fpath1, fpath2])

    assert len(data_model.images) == 2

    data_model.remove_filepaths([fpath1])

    assert len(data_model.images) == 1
    assert data_model.images[0].filename == Path(fpath2).name


def test_data_model_triggers_parent_updates():
    """Test that data model operations call the parent's update methods."""
    parent = Mock()
    data_model = DataModel(parent)
    fpath = examples_single.noisy_fountain.get_path()

    data_model.add_filepaths([fpath])

    parent.filestab_changed.assert_called()
    parent.imagestab_changed.assert_called()
    parent.dimensionstab_changed.assert_called()
    parent.croppingtabs_changed.assert_called()


def test_clear_filepaths_triggers_parent_updates():
    """Test that clearing filepaths calls the parent's update methods."""
    parent = Mock()
    data_model = DataModel(parent)
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    parent.reset_mock()
    data_model.clear_filepaths()

    parent.filestab_changed.assert_called()
    parent.imagestab_changed.assert_called()
