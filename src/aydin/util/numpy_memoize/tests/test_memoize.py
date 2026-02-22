"""Tests for the NumPy-aware last-call memoization decorator."""

import numpy

from aydin.util.numpy_memoize.memoize import memoize_last


def test_memoize():
    """Test memoize_last caches results and recomputes on argument change."""

    class Something(object):
        """Helper class to track memoized method call counts."""

        def __init__(self):
            """Initialize with a private value and a call counter."""
            self._private_val = 2
            self.ncalls = 0

        @memoize_last
        def f(self, x, k, *args, **kwargs):
            "This is a documented method."
            print('private val = ', self._private_val)  # Check access to attributes.
            self.ncalls += 1
            return numpy.random.random()

    something = Something()
    e = numpy.ones(5)
    print('First call:')
    val = something.f(e, 1, thingy='some stuff', z=3, y=-1, extra_arg='not important')
    print('return value = ', val)

    print('Second call (all same args):')

    val = something.f(e, 1, thingy='some stuff', z=3, y=-1, extra_arg='not important')
    print('return value = ', val)

    'Third call (first arg is different):'
    val = something.f(
        2 * e, 1, thingy='some stuff', z=3, y=-1, extra_arg='not important'
    )
    print('return value = ', val)

    print('')
    print('Number of evaluations: ', something.ncalls)
    assert something.ncalls == 2
