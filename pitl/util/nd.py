from random import shuffle


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

def nd_loop(stops):
    if not stops:
        yield ()
        return
    for outer in nd_loop(stops[:-1]):
        for inner in range(0, stops[-1]):
            yield outer + (inner,)

def nd_split_slices(array_shape, nb_slices, do_shuffle=False, margins=None):
    if not array_shape:
        yield ()
        return

    if margins is None:
        margins = (0,)*len(array_shape)

    dim_width = array_shape[-1]

    for outer in nd_split_slices(array_shape[:-1], nb_slices[:-1], do_shuffle=do_shuffle, margins=margins[:-1]):

        n = nb_slices[-1]
        slice_width = int(round(dim_width/n))
        slice_margin=margins[-1]

        slice_start_range = list(range(0, dim_width, slice_width))

        if do_shuffle:
            shuffle(slice_start_range)

        for slice_start in slice_start_range:

            slice_start = max(0,slice_start-slice_margin)
            slice_stop = min(slice_start+slice_width+slice_margin,dim_width)
            yield outer + (slice(slice_start,slice_stop,1),)

def remove_margin_slice(array_shape, slice_with_margin, slice_without_margin):

    slice_tuple = tuple(slice(max(0,v.start-u.start), min(v.stop-u.start,l), 1) for l,u,v in zip(array_shape, slice_with_margin,slice_without_margin))
    return slice_tuple
