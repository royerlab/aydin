"""Outer product and sum operations for n-dimensional arrays."""

import numpy


def outer_product(*arrays):
    """Compute the element-wise outer product of multiple 1D arrays.

    Constructs an n-dimensional array where each element is the product
    of corresponding elements from the input arrays, broadcast across
    all dimensions.

    Parameters
    ----------
    *arrays : numpy.ndarray
        One or more 1D arrays to compute the outer product of.

    Returns
    -------
    numpy.ndarray
        N-dimensional array containing the outer product, where N is
        the number of input arrays.
    """

    outer = numpy.ix_(*arrays)
    shapes = tuple(m.shape for m in outer)
    shape = numpy.max(numpy.array(shapes), axis=0)

    result = numpy.ones(shape, dtype=arrays[0].dtype)
    for coordinate_map in outer:
        result *= numpy.array(coordinate_map)
    return result


def outer_sum(*arrays):
    """Compute the element-wise outer sum of multiple 1D arrays.

    Constructs an n-dimensional array where each element is the sum
    of corresponding elements from the input arrays, broadcast across
    all dimensions.

    Parameters
    ----------
    *arrays : numpy.ndarray
        One or more 1D arrays to compute the outer sum of.

    Returns
    -------
    numpy.ndarray
        N-dimensional array containing the outer sum, where N is
        the number of input arrays.
    """

    outer = numpy.ix_(*arrays)
    shapes = tuple(m.shape for m in outer)
    shape = numpy.max(numpy.array(shapes), axis=0)

    result = numpy.zeros(shape, dtype=arrays[0].dtype)
    for coordinate_map in outer:
        result += numpy.array(coordinate_map)
    return result
