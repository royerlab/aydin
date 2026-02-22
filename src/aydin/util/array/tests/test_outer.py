"""Tests for outer sum and outer product array operations."""

from pprint import pprint

import numpy
import pytest

from aydin.util.array.outer import outer_product, outer_sum


@pytest.fixture(scope="session")
def outer_test_data():
    """Provide two 1D arrays for outer operation tests."""
    u = numpy.linspace(0, 1, 4)
    v = numpy.linspace(-1, 1, 5) ** 2
    return u, v


def test_outer_sum(outer_test_data):
    """Test outer_sum computes element-wise u[i] + v[j]."""
    u, v = outer_test_data
    p = outer_sum(u, v)
    print("")
    pprint(p)

    assert p[2, 3] == u[2] + v[3]


def test_outer_product(outer_test_data):
    """Test outer_product computes element-wise u[i] * v[j]."""
    u, v = outer_test_data
    p = outer_product(u, v)
    print("")
    pprint(p)

    assert p[2, 3] == u[2] * v[3]
