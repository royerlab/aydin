from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io.datasets import examples_single


def test_dimension_analysis():

    image = examples_single.maitre_mouse.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(image)

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 2
    assert 0 in batch_axes and 1 in batch_axes
