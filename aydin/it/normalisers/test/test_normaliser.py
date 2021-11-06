from os import path

import numpy
from numpy import percentile

from aydin.io import imread
from aydin.io.datasets import examples_single
from aydin.it.normalisers.minmax import MinMaxNormaliser
from aydin.it.normalisers.percentile import PercentileNormaliser
from aydin.it.normalisers.shape import ShapeNormaliser


def test_shape_norm_with_channels():
    array = numpy.random.rand(2, 3, 7, 4, 5, 10, 6)

    batch_dims = [False, True, False, True, False, False, False]
    channel_dims = [True, False, True, False, False, False, True]

    norm = ShapeNormaliser()
    (norm_array, axes_permutation, permutated_image_shape) = norm.shape_normalize(
        array, batch_dims, channel_dims
    )

    assert len(norm_array.shape) == 4
    assert norm_array.shape[0] == 3 * 4
    assert norm_array.shape[1] == 2 * 7 * 6

    denorm_array = norm.shape_denormalize(
        norm_array.copy(), axes_permutation, permutated_image_shape
    )

    assert (array == denorm_array).all()


def test_shape_norm_without_channels():
    array = numpy.random.rand(2, 3, 7, 4, 5)

    batch_dims = [False, True, False, True, False]

    norm = ShapeNormaliser()
    (norm_array, axes_permutation, permutated_image_shape) = norm.shape_normalize(
        array, batch_dims
    )

    assert len(norm_array.shape) == 5
    assert norm_array.shape[0] == 3 * 4
    assert norm_array.shape[1] == 1

    denorm_array = norm.shape_denormalize(
        norm_array.copy(), axes_permutation, permutated_image_shape
    )

    assert (array == denorm_array).all()


def test_shape_norm_with_singleton_dim():
    array = numpy.random.rand(2, 3, 1, 4, 5)

    batch_dims = [False, True, False, True, False]

    norm = ShapeNormaliser()
    (norm_array, axes_permutation, permutated_image_shape) = norm.shape_normalize(
        array, batch_dims
    )

    assert len(norm_array.shape) == 4
    assert norm_array.shape[0] == 3 * 4
    assert norm_array.shape[1] == 1

    denorm_array = norm.shape_denormalize(
        norm_array.copy(), axes_permutation, permutated_image_shape
    )

    assert (array == denorm_array).all()


def test_percentile_normaliser():
    input_path = examples_single.fountain.get_path()

    assert path.exists(input_path)

    _test_percentile_normaliser_internal(input_path)


def _test_percentile_normaliser_internal(input_path):
    array, metadata = imread(input_path)
    print(array.shape)

    assert array.dtype == numpy.uint8

    percent = 0.001
    normaliser = PercentileNormaliser(percentile=percent)
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    assert 0.0 <= normalised_array.min() and normalised_array.max() <= 1.0

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint8

    rmin = percentile(denormalised_array, 100 * percent)
    rmax = percentile(denormalised_array, 100 - 100 * percent)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 20


def test_minmax_normaliser():
    input_path = examples_single.fountain.get_path()

    assert path.exists(input_path)

    _test_minmax_normaliser_internal(input_path)


def _test_minmax_normaliser_internal(input_path):
    array, metadata = imread(input_path)
    array = array[0]
    print(array.shape)

    assert array.dtype == numpy.uint8

    normaliser = MinMaxNormaliser()
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint8

    rmin = numpy.min(denormalised_array)
    rmax = numpy.max(denormalised_array)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 5
