import numpy

from aydin.util.offcore.offcore import offcore_array


def test_offcore_array():

    array = offcore_array(shape=(1000, 1000), dtype=numpy.float32)

    vector = numpy.random.rand(1000).astype(numpy.float32)

    for i in range(1000):
        print(i)
        array[:, i] = vector

        assert array[:, i].sum() == vector.sum()

    del array
