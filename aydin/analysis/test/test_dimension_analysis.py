import pytest

from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io import imread
from aydin.io.datasets import examples_single


@pytest.mark.parametrize("image, expected_batch_axes, expected_channel_axes", [
    (examples_single.noisy_newyork.get_array(), [], []),
    (examples_single.maitre_mouse.get_array(), [0, 1], []),
    (examples_single.cognet_nanotube_400fps.get_array()[:, 8:-8, 8:-8], [], []),
    (imread(examples_single.royerlab_hcr.get_path())[0], [], [1]),
    (examples_single.leonetti_snca.get_array(), [], [1])
])
def test_dimension_analysis(image, expected_batch_axes, expected_channel_axes):
    batch_axes, channel_axes = dimension_analysis_on_image(image)

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == len(expected_channel_axes)
    assert len(batch_axes) == len(expected_batch_axes)
    for elem in expected_batch_axes:
        assert elem in batch_axes
    for elem in expected_channel_axes:
        assert elem in channel_axes


@pytest.mark.heavy
def test_dimension_analysis_hela(display: bool = False):

    image = examples_single.hyman_hela.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(image)

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 0
    assert len(batch_axes) == 0


@pytest.mark.unstable
def test_dimension_analysis_flybrain(display: bool = False):

    image_path = examples_single.janelia_flybrain.get_path()
    image, metadata = imread(image_path)

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)
    print(metadata)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 1
    assert 1 in channel_axes
    assert len(batch_axes) == 0
