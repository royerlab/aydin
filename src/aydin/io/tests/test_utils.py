"""Tests for aydin/io/utils.py utility functions."""

import numpy
import pytest

from aydin.io.io import FileMetadata
from aydin.io.utils import (
    get_files_with_most_frequent_extension,
    get_options_json_path,
    get_output_image_path,
    get_save_model_path,
    hyperstack_arrays,
    split_image_channels,
)

# --- Tests for get_output_image_path ---


def test_get_output_image_path_supported_formats(tmp_path):
    """Test get_output_image_path with all supported image formats."""
    formats = [
        '.tif',
        '.tiff',
        '.png',
        '.zarr',
        '.zarr.zip',
        '.czi',
        '.npy',
        '.nd2',
        '.TIF',
    ]

    for fmt in formats:
        input_path = str(tmp_path / f"image{fmt}")
        output_path, counter = get_output_image_path(
            input_path, operation_type="denoised"
        )

        assert output_path is not None
        assert "_denoised" in output_path
        assert counter is None  # No collision expected


def test_get_output_image_path_collision_handling(tmp_path):
    """Test that numeric counter increments when output file already exists."""
    # Create initial output file to cause collision
    input_path = str(tmp_path / "image.tif")
    existing_output = tmp_path / "image_denoised.tif"
    existing_output.touch()

    output_path, counter = get_output_image_path(input_path, operation_type="denoised")

    assert counter == 1
    assert "_denoised1.tif" in output_path


def test_get_output_image_path_multiple_collisions(tmp_path):
    """Test counter increments correctly with multiple existing files."""
    input_path = str(tmp_path / "image.tif")

    # Create multiple existing output files
    (tmp_path / "image_denoised.tif").touch()
    (tmp_path / "image_denoised1.tif").touch()
    (tmp_path / "image_denoised2.tif").touch()

    output_path, counter = get_output_image_path(input_path, operation_type="denoised")

    assert counter == 3
    assert "_denoised3.tif" in output_path


def test_get_output_image_path_output_folder(tmp_path):
    """Test relocation to alternate output folder."""
    input_path = "/some/path/image.tif"
    output_folder = str(tmp_path)

    output_path, counter = get_output_image_path(
        input_path, operation_type="denoised", output_folder=output_folder
    )

    assert str(tmp_path) in output_path
    assert "image_denoised.tif" in output_path
    assert counter is None


def test_get_output_image_path_hyperstacked(tmp_path):
    """Test with hyperstacked operation type."""
    input_path = str(tmp_path / "image.tif")

    output_path, counter = get_output_image_path(
        input_path, operation_type="hyperstacked"
    )

    assert "_hyperstacked" in output_path
    assert counter is None


def test_get_output_image_path_invalid_operation_type(tmp_path):
    """Test ValueError for invalid operation_type."""
    input_path = str(tmp_path / "image.tif")

    with pytest.raises(ValueError, match="invalud value"):
        get_output_image_path(input_path, operation_type="invalid")


def test_get_output_image_path_unsupported_format(tmp_path):
    """Test fallback to .tif for unsupported format."""
    input_path = str(tmp_path / "image.bmp")

    output_path, counter = get_output_image_path(input_path, operation_type="denoised")

    assert output_path.endswith("_denoised.tif")


# --- Tests for get_options_json_path ---


def test_get_options_json_path_basic(tmp_path):
    """Test JSON path generation without collisions."""
    input_path = str(tmp_path / "image.tif")

    options_path = get_options_json_path(input_path)

    assert options_path.endswith("_options.json")
    assert "image_options.json" in options_path


def test_get_options_json_path_collision(tmp_path):
    """Test auto-increment when options file already exists."""
    input_path = str(tmp_path / "image.tif")
    existing_options = tmp_path / "image_options.json"
    existing_options.touch()

    options_path = get_options_json_path(input_path)

    assert "_options1.json" in options_path


def test_get_options_json_path_with_counter(tmp_path):
    """Test with passed_counter from image collision."""
    input_path = str(tmp_path / "image.tif")

    options_path = get_options_json_path(input_path, passed_counter=3)

    assert "_options3.json" in options_path


def test_get_options_json_path_output_folder(tmp_path):
    """Test with alternate output folder."""
    input_path = "/some/path/image.tif"
    output_folder = str(tmp_path)

    options_path = get_options_json_path(input_path, output_folder=output_folder)

    assert str(tmp_path) in options_path


# --- Tests for get_save_model_path ---


def test_get_save_model_path_basic(tmp_path):
    """Test model path generation."""
    input_path = str(tmp_path / "image.tif")

    model_path = get_save_model_path(input_path)

    assert model_path.endswith("_model")
    assert "image_model" in model_path


def test_get_save_model_path_collision(tmp_path):
    """Test auto-increment when model directory already exists."""
    input_path = str(tmp_path / "image.tif")
    existing_model = tmp_path / "image_model"
    existing_model.mkdir()

    model_path = get_save_model_path(input_path)

    assert "_model1" in model_path


def test_get_save_model_path_with_counter(tmp_path):
    """Test with passed_counter coordination."""
    input_path = str(tmp_path / "image.tif")

    model_path = get_save_model_path(input_path, passed_counter=5)

    assert "_model5" in model_path


def test_get_save_model_path_output_folder(tmp_path):
    """Test with alternate output folder."""
    input_path = "/some/path/image.tif"
    output_folder = str(tmp_path)

    model_path = get_save_model_path(input_path, output_folder=output_folder)

    assert str(tmp_path) in model_path


# --- Tests for split_image_channels ---


def _create_metadata_with_channels(shape, axes):
    """Helper to create FileMetadata for channel split tests."""
    metadata = FileMetadata()
    metadata.shape = shape
    metadata.axes = axes
    metadata.batch_axes = tuple(False for _ in shape)
    metadata.channel_axes = tuple(c == 'C' for c in axes)
    return metadata


def test_split_image_channels_multi_channel():
    """Test splitting 3-channel RGB image."""
    # Create a 3-channel image (C, Y, X)
    image_array = numpy.random.rand(3, 64, 64).astype(numpy.float32)
    metadata = _create_metadata_with_channels((3, 64, 64), "CYX")

    result = split_image_channels(image_array, metadata)

    assert result is not None
    splitted_arrays, splitted_metadatas = result

    assert len(splitted_arrays) == 3
    assert len(splitted_metadatas) == 3

    for arr in splitted_arrays:
        assert arr.shape == (64, 64)

    for meta in splitted_metadatas:
        assert meta.axes == "YX"
        assert "C" not in meta.axes
        assert meta.splitted is True


def test_split_image_channels_yx_channel():
    """Test splitting when channel axis is not first."""
    # Create an image with channel in middle (Y, C, X)
    image_array = numpy.random.rand(64, 3, 64).astype(numpy.float32)
    metadata = _create_metadata_with_channels((64, 3, 64), "YCX")

    result = split_image_channels(image_array, metadata)

    assert result is not None
    splitted_arrays, splitted_metadatas = result

    assert len(splitted_arrays) == 3
    for arr in splitted_arrays:
        assert arr.shape == (64, 64)


def test_split_image_channels_no_channel_axis():
    """Test returns None when no 'C' axis present."""
    image_array = numpy.random.rand(64, 64).astype(numpy.float32)
    metadata = _create_metadata_with_channels((64, 64), "YX")

    result = split_image_channels(image_array, metadata)

    assert result is None


def test_split_image_channels_metadata_consistency():
    """Verify metadata is properly updated after split."""
    image_array = numpy.random.rand(3, 64, 64).astype(numpy.float32)
    metadata = _create_metadata_with_channels((3, 64, 64), "CYX")

    splitted_arrays, splitted_metadatas = split_image_channels(image_array, metadata)

    for idx, meta in enumerate(splitted_metadatas):
        assert meta.shape == (64, 64)
        assert len(meta.batch_axes) == 2
        assert len(meta.channel_axes) == 2


# --- Tests for hyperstack_arrays ---


def _create_metadata(shape, axes):
    """Helper to create FileMetadata for hyperstack tests."""
    metadata = FileMetadata()
    metadata.shape = shape
    metadata.axes = axes
    metadata.batch_axes = tuple(False for _ in shape)
    metadata.channel_axes = tuple(False for _ in shape)
    return metadata


def test_hyperstack_arrays_basic():
    """Test stacking 2 same-shape images."""
    arr1 = numpy.random.rand(64, 64).astype(numpy.float32)
    arr2 = numpy.random.rand(64, 64).astype(numpy.float32)
    meta1 = _create_metadata((64, 64), "YX")
    meta2 = _create_metadata((64, 64), "YX")

    stacked_array, stacked_meta = hyperstack_arrays([arr1, arr2], [meta1, meta2])

    assert stacked_array.shape == (2, 64, 64)
    assert stacked_meta.axes == "BYX"
    assert stacked_meta.shape == (2, 64, 64)
    assert stacked_meta.batch_axes == (True, False, False)
    assert stacked_meta.channel_axes == (False, False, False)


def test_hyperstack_arrays_single_image():
    """Test returns unchanged for single image."""
    arr = numpy.random.rand(64, 64).astype(numpy.float32)
    meta = _create_metadata((64, 64), "YX")

    result_arrays, result_metas = hyperstack_arrays([arr], [meta])

    # Should return original list unchanged
    assert result_arrays == [arr]
    assert result_metas == [meta]


def test_hyperstack_arrays_shape_mismatch():
    """Test raises Exception for mismatched shapes."""
    arr1 = numpy.random.rand(64, 64).astype(numpy.float32)
    arr2 = numpy.random.rand(32, 32).astype(numpy.float32)
    meta1 = _create_metadata((64, 64), "YX")
    meta2 = _create_metadata((32, 32), "YX")

    with pytest.raises(Exception, match="not same shape"):
        hyperstack_arrays([arr1, arr2], [meta1, meta2])


def test_hyperstack_arrays_multiple_images():
    """Test stacking more than 2 images."""
    arrays = [numpy.random.rand(32, 32).astype(numpy.float32) for _ in range(5)]
    metas = [_create_metadata((32, 32), "YX") for _ in range(5)]

    stacked_array, stacked_meta = hyperstack_arrays(arrays, metas)

    assert stacked_array.shape == (5, 32, 32)
    assert stacked_meta.shape == (5, 32, 32)


# --- Tests for get_files_with_most_frequent_extension ---


def test_get_files_with_most_frequent_extension(tmp_path):
    """Test directory scanning for most common extension."""
    # Create files with different extensions
    (tmp_path / "image1.tif").touch()
    (tmp_path / "image2.tif").touch()
    (tmp_path / "image3.tif").touch()
    (tmp_path / "data.csv").touch()
    (tmp_path / "readme.txt").touch()

    files = get_files_with_most_frequent_extension(str(tmp_path))

    assert len(files) == 3
    assert all(f.endswith(".tif") for f in files)


def test_get_files_with_most_frequent_extension_tie(tmp_path):
    """Test behavior when extensions have same count."""
    # Create equal number of tif and png files
    (tmp_path / "image1.tif").touch()
    (tmp_path / "image2.tif").touch()
    (tmp_path / "image1.png").touch()
    (tmp_path / "image2.png").touch()

    files = get_files_with_most_frequent_extension(str(tmp_path))

    # Should return files with one of the most frequent extensions
    assert len(files) == 2
    assert all(f.endswith((".tif", ".png")) for f in files)


def test_get_files_with_most_frequent_extension_single_type(tmp_path):
    """Test with only one file type."""
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()

    files = get_files_with_most_frequent_extension(str(tmp_path))

    assert len(files) == 2
    assert all(f.endswith(".png") for f in files)
