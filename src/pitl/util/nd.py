def nd_range(start, stop, dims):
    if not dims:
        yield ()
        return
    for outer in nd_range(start, stop, dims - 1):
        for inner in range(start, stop):
            yield outer + (inner,)


def nd_range_bool_tuple(start, stop, dims):
    if len(dims) == 0:
        yield ()
        return
    for outer in nd_range_bool_tuple(start, stop, dims[:-1]):
        if dims[-1]:
            for inner in range(start, stop):
                yield outer + (inner,)
        else:
            yield outer + ((start+stop)//2,)
