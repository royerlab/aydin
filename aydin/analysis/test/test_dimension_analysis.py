from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io.datasets import examples_single


def test_dimension_analysis():

    image = examples_single.maitre_mouse.get_array()

    dimension_analysis_on_image(image)
