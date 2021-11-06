import numpy


def outer_product(*arrays):

    outer = numpy.ix_(*arrays)
    shapes = tuple(m.shape for m in outer)
    shape = numpy.max(numpy.array(shapes), axis=0)

    result = numpy.ones(shape, dtype=arrays[0].dtype)
    for coordinate_map in outer:
        result *= numpy.array(coordinate_map)
    return result


def outer_sum(*arrays):

    outer = numpy.ix_(*arrays)
    shapes = tuple(m.shape for m in outer)
    shape = numpy.max(numpy.array(shapes), axis=0)

    result = numpy.zeros(shape, dtype=arrays[0].dtype)
    for coordinate_map in outer:
        result += numpy.array(coordinate_map)
    return result
