from aydin.analysis.correlation import correlation_distance
from aydin.io import io
from aydin.io.datasets import examples_single


def demo_analysis():
    for example in examples_single:
        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)
        print(f"File        :  {example}")
        print(f"Metadata    :  {metadata}")
        print(f"Array shape :  {array.shape}")
        print(f"Array dtype :  {array.dtype}")

        correlations = correlation_distance(array)
        print(f"Correlations:  {correlations} ")


demo_analysis()
