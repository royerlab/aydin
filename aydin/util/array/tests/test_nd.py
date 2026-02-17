"""Tests for the n-dimensional array utility functions."""

import numpy
from skimage.data import binary_blobs

from aydin.util.array.nd import (
    extract_tiles,
    nd_loop,
    nd_range,
    nd_range_bool_tuple,
    nd_range_radii,
    nd_split_slices,
    remove_margin_slice,
)


def test_nd_range():
    """Test nd_range generates correct n-dimensional coordinate tuples."""
    print(str(list(nd_range(-1, 1, 3))))
    assert str(list(nd_range(-1, 1, 3))) == (
        "[(-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (-1, 0, 0),"
        " (0, -1, -1), (0, -1, 0), (0, 0, -1), (0, 0, 0)]"
    )


def test_nd_range_radii():
    """Test nd_range_radii generates coordinates within given radii."""
    radii = (1, 2)

    print(str(list(nd_range_radii(radii))))

    assert str(list(nd_range_radii(radii))) == (
        "[(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),"
        " (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),"
        " (1, -2), (1, -1), (1, 0), (1, 1), (1, 2)]"
    )


def test_nd_range_bool_tuple():
    """Test nd_range_bool_tuple respects per-axis enable flags."""
    print(str(list(nd_range_bool_tuple(-1, 1, (True, True, False)))))
    assert (
        str(list(nd_range_bool_tuple(-1, 1, (True, True, False))))
        == "[(-1, -1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 0)]"
    )

    print(str(list(nd_range_bool_tuple(-1, 1, (False, True, False)))))
    assert (
        str(list(nd_range_bool_tuple(-1, 1, (False, True, False))))
        == "[(0, -1, 0), (0, 0, 0)]"
    )


def test_nd_loop():
    """Test nd_loop iterates over all index combinations for given shape."""
    loop_tuple = (2, 1, 3, 5)

    print(str(list(nd_loop(loop_tuple))))
    assert str(list(nd_loop(loop_tuple))) == (
        "[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),"
        " (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 1, 0),"
        " (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3),"
        " (0, 0, 1, 4), (0, 0, 2, 0), (0, 0, 2, 1),"
        " (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4),"
        " (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2),"
        " (1, 0, 0, 3), (1, 0, 0, 4), (1, 0, 1, 0),"
        " (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 1, 3),"
        " (1, 0, 1, 4), (1, 0, 2, 0), (1, 0, 2, 1),"
        " (1, 0, 2, 2), (1, 0, 2, 3), (1, 0, 2, 4)]"
    )


def test_nd_split_slices():
    """Test nd_split_slices covers the full array without gaps or overlaps."""
    array_shape = (96, 17, 117, 45)
    nb_slices = (5, 3, 7, 5)

    slice_tuple_list = list(nd_split_slices(array_shape, nb_slices, do_shuffle=True))

    print(str(slice_tuple_list))

    array_source = numpy.random.choice(a=[False, True], size=array_shape)
    array_target = numpy.random.choice(a=[False, True], size=array_shape)

    for slice_tuple in slice_tuple_list:

        array_target[slice_tuple] = array_source[slice_tuple]
        pass

    assert numpy.all(array_source == array_target)


def test_nd_split_slices_with_margins():
    """Test nd_split_slices with margins and margin removal round-trips correctly."""
    array_shape = (96, 17, 117, 45)
    nb_slices = (5, 3, 7, 5)
    margins = (4, 5, 6, 7)

    slice_tuple_list = list(nd_split_slices(array_shape, nb_slices, do_shuffle=False))
    slice_margin_tuple_list = list(
        nd_split_slices(array_shape, nb_slices, do_shuffle=False, margins=margins)
    )

    # print(str(slice_tuple_list))

    array_source = numpy.random.choice(a=[0, 1], size=array_shape)
    array_target = numpy.zeros(array_shape, dtype=numpy.int64)

    for slice_tuple, slice_margin_tuple in zip(
        slice_tuple_list, slice_margin_tuple_list
    ):

        sliced_array_with_margin = array_source[slice_margin_tuple]
        remove_margin_slice_tuple = remove_margin_slice(
            array_shape, slice_margin_tuple, slice_tuple
        )
        sliced_array_removed_margin = sliced_array_with_margin[
            remove_margin_slice_tuple
        ]
        # print(array_target[slice_tuple].shape)
        # print(sliced_array_removed_margin.shape)
        array_target[slice_tuple] += sliced_array_removed_margin

    assert numpy.all(array_source == array_target)


def test_extract_tiles():
    """Test extract_tiles produces tiles from a 2D binary blob image."""
    blob_image = binary_blobs(length=512, rng=1, n_dim=2)

    tiles = extract_tiles(blob_image, tile_size=64, extraction_step=64, flatten=True)

    print(tiles.shape)
    print(len(tiles))
