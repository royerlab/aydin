import numpy
from napari.util import app_context
from skimage.exposure import rescale_intensity

from pitl.io import io
from pitl.io.examples import example_datasets


def demo_examples():
    """
        ....
    """

    for example in example_datasets.get_list():
        example_file_path = example_datasets.get_path(*example)

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        from napari import ViewerApp
        with app_context():
            viewer = ViewerApp()
            viewer.add_image(array, name='image')


demo_examples()
