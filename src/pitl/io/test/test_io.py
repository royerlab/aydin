from pitl.io import io
from pitl.io.examples import example_datasets


def test_all_download():

    for example in example_datasets.get_list():

        example_file_path = example_datasets.get_path(*example)

        #print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        if not array is None:
            print(f"dataset: {example[1]}, shape:{array.shape}, dtype:{array.dtype} ")
        else:
            print(f"Cannot open dataset: {example[1]}")


