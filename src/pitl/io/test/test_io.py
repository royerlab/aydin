from pitl.io import io
from pitl.io.download_examples import examples


def test_all_download():

    for example in examples.get_list():


        example_file_path = examples.get_path(*example)

        print(f"Trying to open and make sense of file {example_file_path}")

        dataset = io.imread(example_file_path)

        print(dataset)


