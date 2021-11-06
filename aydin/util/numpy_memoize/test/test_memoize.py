import numpy

from aydin.util.numpy_memoize.memoize import memoize_last


def test_memoize():
    class Something(object):
        def __init__(self):
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
