"""Tests for blind spot analysis auto-detection."""

import numpy
import pytest
import scipy
from numpy.random import normal

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.io.datasets import add_noise, camera, examples_single, normalise, rgbtest


def _get_tribolium():
    """Load Tribolium dataset, returning None if unavailable."""
    arr = examples_single.myers_tribolium.get_array()
    return arr


def test_blind_spot_analysis_simulated():
    """Test blind spot detection on a simulated correlated-noise image."""
    image = camera()
    image = normalise(image.astype(numpy.float32, copy=False))
    image = add_noise(image)
    kernel = numpy.array([[0.2, 0.6, 0.2]])
    image = scipy.ndimage.convolve(image, kernel, mode='mirror')

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1), (0, 0), (0, 1)]


def test_blind_spot_analysis_2D_RGB():
    """Test blind spot detection on a 2D RGB image with channel axis."""
    image = rgbtest().astype(numpy.float32)
    image += normal(0, 0.1, image.shape)

    blind_spots, _ = auto_detect_blindspots(image, channel_axes=(False, False, True))
    print(blind_spots)

    # TODO: this corelation is suspicious:
    assert blind_spots == [(-3, 0), (0, 0), (3, 0)]


def test_blind_spot_analysis_tribolium_2D():
    """Test blind spot detection on a single 2D slice of Tribolium data."""

    image = _get_tribolium()
    if image is None:
        pytest.skip("myers_tribolium example could not be loaded")
    image = image[20]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(-1, 0), (0, 0), (1, 0)]


def test_blind_spot_analysis_tribolium_3D():
    """Test blind spot detection on the full 3D Tribolium volume."""
    image = _get_tribolium()
    if image is None:
        pytest.skip("myers_tribolium example could not be loaded")

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_shallow():
    """Test blind spot detection on a shallow 3D Tribolium sub-volume."""
    image = _get_tribolium()
    if image is None:
        pytest.skip("myers_tribolium example could not be loaded")
    image = image[10:-10]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_very_shallow():
    """Test blind spot detection on a very shallow (4-slice) Tribolium sub-volume."""
    image = _get_tribolium()
    if image is None:
        pytest.skip("myers_tribolium example could not be loaded")
    image = image[0:4]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_super_shallow():
    """Test blind spot detection on a single slice extracted from the 3D Tribolium volume."""
    image = _get_tribolium()
    if image is None:
        pytest.skip("myers_tribolium example could not be loaded")
    image = image[0]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    # NOTE: this might not be the right thing to do...
    assert blind_spots == [(-1, 0), (0, 0), (1, 0)]
