"""Demo script for listing, downloading, and viewing example datasets."""

import napari

from aydin.io import io
from aydin.io.datasets import (
    datasets_folder,
    download_all_examples,
    download_from_zenodo,
    examples_single,
)


def demo_examples_single():
    """Print all available single-image example datasets."""
    for dataset in examples_single:
        print(dataset)


def demo_download():
    """Download a single example dataset (mandrill) from Zenodo."""
    _, filename = examples_single.generic_mandrill.value
    print(download_from_zenodo(filename, datasets_folder))


def demo_all_download():
    """Download all available example datasets."""
    download_all_examples()


def demo_examples():
    """Load and display each example dataset in a napari viewer."""
    for example in examples_single:
        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        viewer = napari.Viewer()
        viewer.add_image(array, name='image')
        napari.run()


if __name__ == "__main__":
    demo_examples()
