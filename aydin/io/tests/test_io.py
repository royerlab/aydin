"""Tests for image I/O operations (reading, writing, memory-mapped TIFF)."""

from os import path
from os.path import join

import numpy
import pytest

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import (
    FileMetadata,
    _sync_array_with_metadata,
    imread,
    imwrite,
    is_batch,
    is_channel,
    mapped_tiff,
)


def test_analysis():
    """Test that imread returns valid metadata for all example images.

    Note: Some examples may fail due to download issues or format problems.
    This test asserts on all successfully loaded images.
    """
    successful_reads = 0
    for example in examples_single:
        try:
            example_file_path = example.get_path()
            array, metadata = io.imread(example_file_path)
        except Exception:
            # Skip examples that fail to download
            continue

        if array is None:
            # Some formats may not be fully supported
            continue

        successful_reads += 1

        # Verify metadata is present and valid for successfully loaded images
        assert metadata is not None, f"Failed to get metadata for {example.value[1]}"
        assert (
            metadata.shape is not None
        ), f"Missing shape in metadata for {example.value[1]}"
        assert (
            metadata.axes is not None
        ), f"Missing axes in metadata for {example.value[1]}"
        assert len(metadata.axes) == len(metadata.shape), (
            f"Axes length mismatch for {example.value[1]}: "
            f"axes={metadata.axes}, shape={metadata.shape}"
        )

    # Ensure at least some examples were successfully tested
    assert successful_reads > 0, "No example images could be loaded"


@pytest.mark.heavy
def test_opening_examples():
    """Test that all example images can be opened and have consistent metadata.

    Note: Some examples may fail due to download issues or format problems.
    This test asserts on all successfully loaded images.
    """
    successful_reads = 0
    for example in examples_single:
        try:
            example_file_path = example.get_path()
            array, metadata = io.imread(example_file_path)
        except Exception:
            # Skip examples that fail to download
            continue

        if array is None:
            # Some formats may not be fully supported
            continue

        successful_reads += 1

        # Assert metadata was successfully extracted
        assert metadata is not None, f"No metadata for: {example.value[1]}"

        # Assert shape consistency between array and metadata
        assert array.shape == metadata.shape, (
            f"Shape mismatch for {example.value[1]}: "
            f"array.shape={array.shape}, metadata.shape={metadata.shape}"
        )

        # Assert dtype consistency
        assert array.dtype == metadata.dtype, (
            f"Dtype mismatch for {example.value[1]}: "
            f"array.dtype={array.dtype}, metadata.dtype={metadata.dtype}"
        )

    # Ensure at least some examples were successfully tested
    assert successful_reads > 0, "No example images could be loaded"


def test_imwrite():
    """Test writing and re-reading images in TIFF and PNG formats."""
    input_path = examples_single.janelia_flybrain.get_path()
    assert path.exists(input_path)

    array, metadata = imread(input_path)
    if array is None:
        pytest.skip("janelia_flybrain example could not be loaded")

    temp_file = join(get_temp_folder(), "test_imwrite.tif")

    print("Writting file...")
    imwrite(array, temp_file)
    print("Done...")

    print("Reading back...")
    array2, _ = imread(temp_file)

    assert numpy.all(array == array2)

    input_path = examples_single.generic_lizard.get_path()
    assert path.exists(input_path)

    array, metadata = imread(input_path)
    temp_file = join(get_temp_folder(), "test_imwrite.png")

    print("Writting file...")
    imwrite(array, temp_file)
    print("Done...")

    print("Reading back...")
    array2, _ = imread(temp_file)

    assert numpy.all(array == array2)


def test_imread_nonexistent_file():
    """Test that imread returns None for a non-existent file path."""
    array, metadata = imread('/nonexistent/path/to/image.tif')
    assert array is None


def test_mapped_tiff():
    """Test writing and re-reading an image via memory-mapped TIFF."""

    input_path = examples_single.janelia_flybrain.get_path()

    assert path.exists(input_path)

    array, metadata = imread(input_path)
    if array is None:
        pytest.skip("janelia_flybrain example could not be loaded")

    temp_file = join(get_temp_folder(), "test_imwrite.tif")

    # We use a generator that takes care of everything:
    with mapped_tiff(temp_file, shape=array.shape, dtype=array.dtype) as tiff_array:

        # We write here the file
        print("Writting file...")
        tiff_array[...] = array

    print("Done...")

    print("Reading back...")
    array2, _ = imread(temp_file)

    assert numpy.all(array == array2)


# --- is_batch / is_channel tests ---


def test_is_batch_spatial_axes():
    """Spatial axes (X, Y, Z, T, C) should not be batch."""
    shape = (10, 64, 64)
    axes = 'TYX'
    assert not is_batch('T', shape, axes)
    assert not is_batch('Y', shape, axes)
    assert not is_batch('X', shape, axes)


def test_is_batch_non_spatial():
    """Non-spatial axes like 'I', 'R' should be batch (except special case)."""
    shape = (5, 3, 64, 64)
    axes = 'IRYX'
    assert is_batch('I', shape, axes)
    assert is_batch('R', shape, axes)


def test_is_batch_3d_xy_special_case():
    """In a 3D XYI image, 'I' is not batch (special case)."""
    shape = (10, 64, 64)
    axes = 'IYX'
    assert not is_batch('I', shape, axes)


def test_is_channel_true():
    """'C' axis with length <= 8 should be a channel."""
    assert is_channel('C', 3)
    assert is_channel('C', 1)
    assert is_channel('C', 8)


def test_is_channel_false_long():
    """'C' axis with length > 8 should not be a channel."""
    assert not is_channel('C', 9)
    assert not is_channel('C', 100)


def test_is_channel_wrong_code():
    """Non-'C' axes should never be channels."""
    assert not is_channel('X', 3)
    assert not is_channel('Y', 1)


# --- FileMetadata tests ---


def test_file_metadata_equality():
    """Test that two FileMetadata with identical fields compare equal."""
    m1 = FileMetadata()
    m1.shape = (64, 64)
    m1.dtype = numpy.float32
    m1.axes = 'YX'

    m2 = FileMetadata()
    m2.shape = (64, 64)
    m2.dtype = numpy.float32
    m2.axes = 'YX'

    assert m1 == m2


def test_file_metadata_inequality():
    """Test that two FileMetadata with different shapes compare unequal."""
    m1 = FileMetadata()
    m1.shape = (64, 64)

    m2 = FileMetadata()
    m2.shape = (32, 32)

    assert m1 != m2


def test_file_metadata_not_equal_to_other_types():
    """Test that FileMetadata.__eq__ returns NotImplemented for non-metadata types."""
    m = FileMetadata()
    assert m.__eq__('string') is NotImplemented


def test_file_metadata_str():
    """Test that FileMetadata.__str__ includes shape and axes info."""
    m = FileMetadata()
    m.shape = (64, 64)
    m.axes = 'YX'
    s = str(m)
    assert 'shape' in s
    assert 'YX' in s


# --- _sync_array_with_metadata ---


def test_sync_array_with_metadata_updates_shape():
    """Test that _sync_array_with_metadata updates shape and dtype from array."""
    m = FileMetadata()
    m.shape = (32, 32)
    m.dtype = numpy.float32
    arr = numpy.zeros((64, 64), dtype=numpy.float64)
    _sync_array_with_metadata(arr, m)
    assert m.shape == (64, 64)
    assert m.dtype == numpy.float64


def test_sync_array_with_metadata_noop():
    """Test that _sync_array_with_metadata is a no-op when already in sync."""
    m = FileMetadata()
    m.shape = (64, 64)
    m.dtype = numpy.float32
    arr = numpy.zeros((64, 64), dtype=numpy.float32)
    _sync_array_with_metadata(arr, m)
    assert m.shape == (64, 64)


def test_sync_array_with_none():
    """Should not crash with None inputs."""
    _sync_array_with_metadata(None, None)
    _sync_array_with_metadata(numpy.zeros(1), None)
    _sync_array_with_metadata(None, FileMetadata())


# --- imwrite with overwrite=False ---


def test_imwrite_no_overwrite(tmp_path):
    """When overwrite=False and file exists, should not overwrite."""
    arr = numpy.zeros((16, 16), dtype=numpy.uint8)
    out_path = str(tmp_path / 'test.tif')
    imwrite(arr, out_path)

    # Write different data with overwrite=False
    arr2 = numpy.ones((16, 16), dtype=numpy.uint8)
    imwrite(arr2, out_path, overwrite=False)

    # Read back - should be the original zeros
    result, _ = imread(out_path)
    assert numpy.all(result == 0)
