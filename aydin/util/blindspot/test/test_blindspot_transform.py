from pprint import pprint

import numpy

from aydin.util.blindspot.blindspot_transform import BlindSpotTransform


def test_blindspot_transform():

    blind_spots = [(0, -1), (0, 0), (0, 1)]
    bst = BlindSpotTransform(blind_spots)

    array = numpy.arange(12).reshape(2, 6)

    t_input_array, t_target_array = bst.transform(array, array)

    # We fake perfect inference here:
    t_translated_array = t_target_array

    it_array = bst.inverse_transform(t_translated_array)

    pprint(array)
    pprint(t_input_array)
    pprint(t_target_array)
    pprint(it_array)

    assert (it_array == array).all()
