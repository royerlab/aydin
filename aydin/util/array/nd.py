import numbers
from math import ceil
from random import shuffle
import numpy
from numpy.lib.stride_tricks import as_strided


def nd_range(start, stop, dims):
    """
    nd_range

    Parameters
    ----------
    start : int
    stop : int
    dims : int

    """
    if not dims:
        yield ()
        return
    for outer in nd_range(start, stop, dims - 1):
        for inner in range(start, stop):
            yield outer + (inner,)


def nd_range_radii(radii):
    """
    nd_range_radii

    Parameters
    ----------
    radii : list

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
    """
    nd_range_bool_tuple

    Parameters
    ----------
    start : int
    stop : int
    dims : int

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
    """
    nd_loop

    Parameters
    ----------
    stops : list

    """
    if not stops:
        yield ()
        return
    for outer in nd_loop(stops[:-1]):
        for inner in range(0, stops[-1]):
            yield outer + (inner,)


def nd_split_slices(array_shape, nb_slices, do_shuffle=False, margins=None):
    """
    nd_split_slices

    Parameters
    ----------
    array_shape : tuple
    nb_slices : list
    do_shuffle : tuple
    margins : tuple

    Returns
    -------

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
    """
    Remove_margin_slice

    Parameters
    ----------
    array_shape : tuple
    slice_with_margin : array_like
    slice_without_margin : array_like

    Returns
    -------
    sliced_tuple : tuple

    """
    slice_tuple = tuple(
        slice(max(0, v.start - u.start), min(v.stop - u.start, l), 1)
        for l, u, v in zip(array_shape, slice_with_margin, slice_without_margin)
    )
    return slice_tuple


def extract_tiles(arr, tile_size=8, extraction_step=1, flatten=False):
    """
    Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    tile_size : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
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
