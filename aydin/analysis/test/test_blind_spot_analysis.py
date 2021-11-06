import numpy
import scipy
from numpy.random import normal

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.io.datasets import camera, normalise, add_noise, rgbtest, examples_single


def test_blind_spot_analysis_simulated():
    image = camera()
    image = normalise(image.astype(numpy.float32, copy=False))
    image = add_noise(image)
    kernel = numpy.array([[0.2, 0.6, 0.2]])
    image = scipy.ndimage.convolve(image, kernel, mode='mirror')

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1), (0, 0), (0, 1)]


def test_blind_spot_analysis_2D_RGB():
    image = rgbtest().astype(numpy.float32)
    image += normal(0, 0.1, image.shape)

    blind_spots, _ = auto_detect_blindspots(image, channel_axes=(False, False, True))
    print(blind_spots)

    # TODO: this corelation is suspicious:
    assert blind_spots == [(-3, 0), (0, 0), (3, 0)]


def test_blind_spot_analysis_tribolium_2D():

    image = examples_single.myers_tribolium.get_array()
    image = image[20]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(-1, 0), (0, 0), (1, 0)]


def test_blind_spot_analysis_tribolium_3D():
    image = examples_single.myers_tribolium.get_array()

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_shallow():
    image = examples_single.myers_tribolium.get_array()
    image = image[10:-10]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_very_shallow():
    image = examples_single.myers_tribolium.get_array()
    image = image[0:4]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    assert blind_spots == [(0, -1, 0), (0, 0, 0), (0, 1, 0)]


def test_blind_spot_analysis_tribolium_3D_super_shallow():
    image = examples_single.myers_tribolium.get_array()
    image = image[0]

    blind_spots, _ = auto_detect_blindspots(image)
    print(blind_spots)

    # NOTE: this might not be the right thing to do...
    assert blind_spots == [(-1, 0), (0, 0), (1, 0)]
