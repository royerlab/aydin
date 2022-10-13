import pytest

from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io.datasets import examples_single


def test_dimension_analysis_noisynewyork(display: bool = False):

    image = examples_single.noisy_newyork.get_array()

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


def test_dimension_analysis_maitre(display: bool = False):

    image = examples_single.maitre_mouse.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(image)

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 0
    assert len(batch_axes) == 2
    assert 0 in batch_axes and 1 in batch_axes


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


def test_dimension_analysis_cognet(display: bool = False):

    image = examples_single.cognet_nanotube_400fps.get_array()

    # we remove some weird pixels:
    image = image[:, 8:-8, 8:-8]

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


# def test_dimension_analysis_huang(display: bool = False):
#
#     image = examples_single.huang_fixed_pattern_noise.get_array()
#
#     batch_axes, channel_axes = dimension_analysis_on_image(image)
#
#     print(batch_axes)
#     print(channel_axes)
#
#     if display:
#         import napari
#
#         viewer = napari.Viewer()
#         viewer.add_image(image, name='image')
#         napari.run()
#
#     assert len(channel_axes) == 0
#     assert len(batch_axes) == 1
#     assert 0 in batch_axes


def test_dimension_analysis_royer(display: bool = False):

    image = examples_single.royerlab_hcr.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 1
    assert 1 in channel_axes
    assert len(batch_axes) == 0


def test_dimension_analysis_flybrain(display: bool = False):

    image = examples_single.janelia_flybrain.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 1
    assert 1 in channel_axes
    assert len(batch_axes) == 0


def test_dimension_analysis_leonetti(display: bool = False):

    image = examples_single.leonetti_snca.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 1
    assert 1 in channel_axes
    assert len(batch_axes) == 0


def test_dimension_analysis_myers(display: bool = False):

    image = examples_single.myers_tribolium.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        napari.run()

    assert len(channel_axes) == 0
    assert len(batch_axes) == 0
