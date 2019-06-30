from pitl.io import io
from pitl.io.datasets import examples_single


def test_all_download():

    for example in examples_single.get_list():

        example_file_path = examples_single.get_path(*example)

        #print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        if not array is None:
            print(f"dataset: {example[1]}, shape:{array.shape}, dtype:{array.dtype} ")
        else:
            print(f"Cannot open dataset: {example[1]}")


