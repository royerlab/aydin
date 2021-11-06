from aydin.util.misc.combinatorics import closest_product


def test_closest_product():

    u = [1, 2, 5, 7, 9, 10]

    N = 15
    result = closest_product(u, N)
    print(f"closest_product({u}, {N}) = {result}")
    assert result == [1, 3]

    N = 27
    result = closest_product(u, N)
    print(f"closest_product({u}, {N}) = {result}")
    assert result is None
