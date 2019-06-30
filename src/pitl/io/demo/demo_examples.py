from napari.util import app_context

from pitl.io import io
from pitl.io.datasets import examples_single


def demo_examples():
    """
        ....
    """

    for example in examples_single.get_list():
        example_file_path = examples_single.get_path(*example)

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        from napari import ViewerApp
        with app_context():
            viewer = ViewerApp()
            viewer.add_image(array, name='image')


demo_examples()
