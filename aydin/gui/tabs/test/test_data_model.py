"""Tests for the DataModel class."""

from unittest.mock import Mock

import numpy
import pytest

from aydin.gui.tabs.data_model import DataModel
from aydin.io import imread
from aydin.io.datasets import examples_single
from aydin.io.io import FileMetadata


def test_not_adding_non_existing_filepaths():
    data_model = DataModel(Mock())
    data_model.add_filepaths(["/asdfas/asdfasd/fdsaf/fas", "/fdsa/fdsa/fdsa/fdsa"])

    assert data_model.filepaths == {}


def test_adding_files():
    data_model = DataModel(Mock())
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
    data_model = DataModel(Mock())
    fpath = examples_single.noisy_fountain.get_path()
    data_model.add_filepaths([fpath])

    data_model.clear_filepaths()

    assert data_model.filepaths == dict()


def test_removing_multiple_but_not_all_files():
    data_model = DataModel(Mock())
    fpath1 = examples_single.noisy_fountain.get_path()
    fpath2 = examples_single.generic_camera.get_path()
    data_model.add_filepaths([fpath1, fpath2])

    assert len(data_model.filepaths) == 2

    data_model.remove_filepaths([fpath1])
    assert len(data_model.filepaths) == 1
    assert fpath2 in data_model.filepaths
