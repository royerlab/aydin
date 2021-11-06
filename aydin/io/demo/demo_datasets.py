import napari

from aydin.io import io
from aydin.io.datasets import (
    download_from_gdrive,
    examples_single,
    download_all_examples,
    datasets_folder,
)


def demo_examples_single():
    for dataset in examples_single:
        print(dataset)


def demo_download():
    print(
        download_from_gdrive(*examples_single.generic_mandrill.value, datasets_folder)
    )


def demo_all_download():
    download_all_examples()


def demo_examples():
    """
    ....
    """

    for example in examples_single:
        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(array, name='image')


demo_examples()
