"""N-dimensional array utility functions for splitting, iteration, and tiling."""

import numbers
from math import ceil
from random import shuffle

import numpy
from numpy.lib.stride_tricks import as_strided


def nd_range(start, stop, dims):
    """Generate all n-dimensional index tuples within a range.

    Recursively generates all tuples of length ``dims`` where each
    element is in the range [start, stop).

    Parameters
    ----------
    start : int
        Start of range (inclusive).
    stop : int
        End of range (exclusive).
    dims : int
        Number of dimensions (tuple length).

    Yields
    ------
    tuple of int
        Index tuples of length ``dims``.
    """
    if not dims:
        yield ()
        return
    for outer in nd_range(start, stop, dims - 1):
        for inner in range(start, stop):
            yield outer + (inner,)


def nd_range_radii(radii):
    """Generate all n-dimensional index tuples within given radii per axis.

    For each axis, iterates from ``-radius`` to ``+radius`` (inclusive),
    producing all possible coordinate tuples.

    Parameters
    ----------
    radii : list of int
        Radius for each dimension.

    Yields
    ------
    tuple of int
        Index tuples spanning the given radii.
    """
    if not radii:
        yield ()
        return

    for outer in nd_range_radii(radii[:-1]):

        radius = radii[-1]
        start = -radius
        stop = +radius + 1

        for inner in range(start, stop):
            yield outer + (inner,)


def nd_range_bool_tuple(start, stop, dims):
    """Generate n-dimensional index tuples with per-axis enable flags.

    For each dimension, if the corresponding flag in ``dims`` is True,
    iterates over [start, stop). Otherwise, yields only the midpoint.

    Parameters
    ----------
    start : int
        Start of range (inclusive).
    stop : int
        End of range (exclusive).
    dims : tuple of bool
        Boolean flags indicating which dimensions to iterate over.

    Yields
    ------
    tuple of int
        Index tuples respecting the per-axis enable flags.
    """
    if len(dims) == 0:
        yield ()
        return
    for outer in nd_range_bool_tuple(start, stop, dims[:-1]):
        if dims[-1]:
            for inner in range(start, stop):
                yield outer + (inner,)
        else:
            yield outer + ((start + stop) // 2,)


def nd_loop(stops):
    """Generate all n-dimensional index tuples from zero up to given stops.

    Parameters
    ----------
    stops : list of int
        Upper bound (exclusive) for each dimension.

    Yields
    ------
    tuple of int
        Index tuples with each element in range [0, stop) for its dimension.
    """
    if not stops:
        yield ()
        return
    for outer in nd_loop(stops[:-1]):
        for inner in range(0, stops[-1]):
            yield outer + (inner,)


def nd_split_slices(array_shape, nb_slices, do_shuffle=False, margins=None):
    """Generate slice tuples that partition an n-dimensional array.

    Divides each axis into the specified number of slices with optional
    overlap margins, useful for tiled parallel processing.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the array to split.
    nb_slices : tuple of int
        Number of slices along each dimension.
    do_shuffle : bool
        Whether to shuffle the slice order (useful for load balancing).
    margins : tuple of int, optional
        Overlap margin per dimension. Defaults to zero for all dimensions.

    Yields
    ------
    tuple of slice
        Slice tuples that cover the array with the requested partitioning.
    """
    if not array_shape:
        yield ()
        return

    if margins is None:
        margins = (0,) * len(array_shape)

    dim_width = array_shape[-1]

    for outer in nd_split_slices(
        array_shape[:-1], nb_slices[:-1], do_shuffle=do_shuffle, margins=margins[:-1]
    ):

        n = nb_slices[-1]
        slice_width = int(ceil(dim_width / n))
        slice_margin = margins[-1]

        slice_start_range = list(range(0, dim_width, slice_width))

        if do_shuffle:
            shuffle(slice_start_range)

        for slice_start in slice_start_range:

            start = max(0, slice_start - slice_margin)
            stop = min(slice_start + slice_width + slice_margin, dim_width)
            yield outer + (slice(start, stop, 1),)


def remove_margin_slice(array_shape, slice_with_margin, slice_without_margin):
    """Compute a slice that removes the margin from a margined tile.

    Given slice tuples with and without margins, returns a slice into
    the margined tile that extracts only the non-margin region.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the original array.
    slice_with_margin : tuple of slice
        Slice including margins.
    slice_without_margin : tuple of slice
        Slice without margins (target region).

    Returns
    -------
    tuple of slice
        Slice into the margined tile that extracts the non-margin region.
    """
    slice_tuple = tuple(
        slice(max(0, v.start - u.start), min(v.stop - u.start, a), 1)
        for a, u, v in zip(array_shape, slice_with_margin, slice_without_margin)
    )
    return slice_tuple


def extract_tiles(arr, tile_size=8, extraction_step=1, flatten=False):
    """Extract patches from an n-dimensional array using stride tricks.

    Uses NumPy stride tricks for O(1) patch extraction without copying data.
    The resulting array has 2N dimensions: the first N index patch positions
    and the last N contain patch contents.

    Parameters
    ----------
    arr : numpy.ndarray
        N-dimensional array from which to extract patches.
    tile_size : int or tuple of int
        Shape of each patch. If an integer, all dimensions use the same size.
    extraction_step : int or tuple of int
        Step size between patches. If an integer, uniform across all dimensions.
    flatten : bool
        If True, reshape output to (n_patches, *tile_size).

    Returns
    -------
    numpy.ndarray
        Array of extracted patches. If ``flatten`` is False, shape is
        2N-dimensional; otherwise (n_patches, *tile_size).
    """

    arr_ndim = arr.ndim

    if isinstance(tile_size, numbers.Number):
        tile_size = tuple([tile_size] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = numpy.empty(arr.shape).strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = numpy.empty(arr[slices].shape).strides

    patch_indices_shape = (
        (numpy.array(arr.shape) - numpy.array(tile_size))
        // numpy.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(tile_size))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)

    if flatten:
        patches = patches.reshape((-1,) + patches.shape[-arr.ndim :])

    return patches
