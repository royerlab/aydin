from pitl.util.nd import nd_range, nd_range_bool_tuple


def test_nd_range():

    print(str(list(nd_range(-1,1,3))))
    assert str(list(nd_range(-1,1,3)))== "[(-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (-1, 0, 0), (0, -1, -1), (0, -1, 0), (0, 0, -1), (0, 0, 0)]"


def test_nd_range_bool_tuple():

    print(str(list(nd_range_bool_tuple(-1,1,(True, True , False)))))
    assert str(list(nd_range_bool_tuple(-1,1,(True, True , False)))) == "[(-1, -1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 0)]"

    print(str(list(nd_range_bool_tuple(-1, 1, (False, True, False)))))
    assert str(list(nd_range_bool_tuple(-1, 1, (False, True, False)))) == "[(0, -1, 0), (0, 0, 0)]"
