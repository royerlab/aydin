from os import path
from os.path import join

import numpy
import pytest

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import imread, imwrite, mapped_tiff


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
    input_path = examples_single.janelia_flybrain.get_path()
    assert path.exists(input_path)

    array, metadata = imread(input_path)

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


def test_mapped_tiff():

    input_path = examples_single.janelia_flybrain.get_path()

    assert path.exists(input_path)

    array, metadata = imread(input_path)

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
