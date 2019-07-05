from pitl.util.combinatorics import closest_product


def test_closest_product():

    epsilon = 0.1
    u = [1,2,4,7,10]

    N = 15
    result = closest_product(u, N ,epsilon)
    print(f"closest_product({u}, {N} ,{epsilon}) = {result}")
    assert result==[1,3]

    N= 27
    result = closest_product(u, N ,epsilon)
    print(f"closest_product({u}, {N} ,{epsilon}) = {result}")
    assert result==[2,3]

