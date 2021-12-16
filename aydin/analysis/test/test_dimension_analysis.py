from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io.datasets import examples_single


def test_dimension_analysis_maitre():

    image = examples_single.maitre_mouse.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, algorithm='butterworth'
    )

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 2
    assert 0 in batch_axes and 1 in batch_axes


def test_dimension_analysis_hela():

    image = examples_single.hyman_hela.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, algorithm='butterworth'
    )

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 0


def test_dimension_analysis_cognet():

    image = examples_single.cognet_nanotube_400fps.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, algorithm='butterworth'
    )

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 2
    assert 1 in batch_axes


def test_dimension_analysis_huang():

    image = examples_single.huang_fixed_pattern_noise.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, algorithm='butterworth'
    )

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 1
    assert 2 in batch_axes


def test_dimension_analysis_royer():

    image = examples_single.royerlab_hcr.get_array()

    batch_axes, channel_axes = dimension_analysis_on_image(
        image, algorithm='butterworth', max_channels_per_axis=6
    )

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == 0
    assert len(batch_axes) == 0
