from pprint import pprint

import numpy

from aydin.util.array.outer import outer_product, outer_sum


def test_outer_sum():

    u = numpy.linspace(0, 1, 4)
    v = numpy.linspace(-1, 1, 5) ** 2
    p = outer_sum(u, v)
    print("")
    pprint(p)

    assert p[2, 3] == u[2] + v[3]


def test_outer_product():

    u = numpy.linspace(0, 1, 4)
    v = numpy.linspace(-1, 1, 5) ** 2
    p = outer_product(u, v)
    print("")
    pprint(p)

    assert p[2, 3] == u[2] * v[3]
