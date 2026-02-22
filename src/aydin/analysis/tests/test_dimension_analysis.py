"""Tests for image dimension analysis (batch and channel axis detection)."""

import pytest

from aydin.analysis.dimension_analysis import dimension_analysis_on_image
from aydin.io import imread
from aydin.io.datasets import examples_single


def _load_example(example, slicing=None):
    """Load an example image, returning None if unavailable."""
    try:
        arr = example.get_array()
        if arr is not None and slicing is not None:
            arr = arr[slicing]
        return arr
    except Exception:
        return None


def _load_example_imread(example):
    """Load an example image via imread, returning None if unavailable."""
    try:
        path = example.get_path()
        arr, _ = imread(path)
        return arr
    except Exception:
        return None


@pytest.mark.parametrize(
    "example, slicing, expected_batch_axes, expected_channel_axes",
    [
        pytest.param(
            examples_single.noisy_newyork,
            None,
            [],
            [],
            id="newyork",
        ),
        pytest.param(
            examples_single.maitre_mouse,
            None,
            [0, 1],
            [],
            id="mouse",
        ),
        pytest.param(
            examples_single.cognet_nanotube_400fps,
            (slice(None), slice(8, -8), slice(8, -8)),
            [],
            [],
            id="cognet",
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            examples_single.leonetti_snca,
            None,
            [],
            [1],
            id="leonetti",
            marks=pytest.mark.heavy,
        ),
    ],
)
def test_dimension_analysis(
    example, slicing, expected_batch_axes, expected_channel_axes
):
    """Test batch and channel axis detection on various example images."""
    image = _load_example(example, slicing)
    if image is None:
        pytest.skip(f"Example {example} could not be loaded")

    batch_axes, channel_axes = dimension_analysis_on_image(image)

    print(batch_axes)
    print(channel_axes)

    assert len(channel_axes) == len(expected_channel_axes)
    assert len(batch_axes) == len(expected_batch_axes)
    for elem in expected_batch_axes:
        assert elem in batch_axes
    for elem in expected_channel_axes:
        assert elem in channel_axes


def test_dimension_analysis_hcr():
    """Test dimension analysis on HCR image detects one channel axis."""
    image = _load_example_imread(examples_single.royerlab_hcr)
    if image is None:
        pytest.skip("royerlab_hcr example could not be loaded")

    batch_axes, channel_axes = dimension_analysis_on_image(image)

    assert len(channel_axes) == 1
    assert 1 in channel_axes


@pytest.mark.heavy
def test_dimension_analysis_hela(display: bool = False):
    """Test dimension analysis on HeLa cell image has no batch or channel axes."""

    image = examples_single.hyman_hela.get_array()
    if image is None:
        pytest.skip("hyman_hela example could not be loaded")

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


@pytest.mark.heavy
def test_dimension_analysis_flybrain(display: bool = False):
    """Test dimension analysis on fly brain image detects one channel axis."""

    image_path = examples_single.janelia_flybrain.get_path()
    image, metadata = imread(image_path)
    if image is None:
        pytest.skip("janelia_flybrain example could not be loaded")

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
