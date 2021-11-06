from mock import Mock

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
    fpath = examples_single.fountain.get_path()
    data_model.add_filepaths([fpath])
    array, metadata = imread(fpath)
    target_dict = {fpath: (array, metadata)}

    for key in data_model.filepaths.keys():
        for elem1, elem2 in zip(data_model.filepaths[key], target_dict[key]):
            if isinstance(elem1, FileMetadata):
                assert elem1 == elem2
            else:
                assert elem1.all() == elem2.all()


def test_removing_files():
    data_model = DataModel(Mock())
    fpath = examples_single.fountain.get_path()
    data_model.add_filepaths([fpath])

    data_model.clear_filepaths()

    assert data_model.filepaths == dict()


def test_removing_multiple_but_not_all_files():
    pass


def test_hyperstack():
    pass


def test_hyperstack_attempt_images_of_different_shapes():
    pass
