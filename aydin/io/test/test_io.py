import pytest
from os import path
from os.path import join

import numpy

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import imread, mapped_tiff, imwrite


def test_analysis():

    for example in examples_single:

        example_file_path = example.get_path()

        _, analysis_result = io.imread(example_file_path)

        print(analysis_result)


@pytest.mark.heavy
def test_opening_examples():

    for example in examples_single:

        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        if array is not None:
            print(
                f"dataset: {example.value[1]}, shape:{array.shape}, dtype:{array.dtype} "
            )
        else:
            print(f"Cannot open dataset: {example.value[1]}")


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
