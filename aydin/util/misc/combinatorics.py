def closest_product(values, N, tol_min=-0.1, tol_max=0.1):
    """
    Given a list of integers u=[a_0, ..., a_m]  find the set of indices
    {i_0, ..., i_n} for that list such that the product u[i_0]* ... *u[i_n]
    is the closest to a given integer N.

    This method is pyramid heuristic approximation that returns the closest
    solution within N*[tol_min,tol_max]

    Parameters
    ----------
    values : array_like
    N : int
    tol_min : float
    tol_max : float

    Returns
    -------
    List of ints or None

    """
    length = len(values)

    # it is just there...
    if N in values:
        return [values.index(N)]

    # let's look at pairs of values:
    for a in range(0, length):
        for b in range(a + 1, length):
            value = abs(values[a] * values[b] - N) / N
            if tol_min <= value and value < tol_max:
                return [a, b]

    # let's look at triplets of values:
    for a in range(0, length):
        for b in range(a + 1, length):
            for c in range(b + 1, length):
                value = abs(values[a] * values[b] * values[c] - N) / N
                if tol_min <= value and value < tol_max:
                    return [a, b, c]

    # let's look at quadruplets of values:
    for a in range(0, length):
        for b in range(a + 1, length):
            for c in range(b + 1, length):
                for d in range(c + 1, length):
                    value = abs(values[a] * values[b] * values[c] * values[d] - N) / N
                    if tol_min <= value and value < tol_max:
                        return [a, b, c, d]

    # let's look at quintuplets of values:
    for a in range(0, length):
        for b in range(a + 1, length):
            for c in range(b + 1, length):
                for d in range(c + 1, length):
                    for e in range(d + 1, length):
                        value = (
                            abs(
                                values[a]
                                * values[b]
                                * values[c]
                                * values[d]
                                * values[e]
                                - N
                            )
                            / N
                        )
                        if tol_min <= value and value < tol_max:
                            return [a, b, c, d, e]

    # There is no chance we need more...

    # Given the constraints, it might not be possible to find a soliution anyway.
    # in that case we give up and return None:
    return None
