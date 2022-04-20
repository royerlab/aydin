import numpy


def bbox_idx(x, thresh=None):
    """
    Returns indices of bounding box of receptive filed. This is useful for visualize receptive field.

    Parameters
    ----------
    x
        output image of the target model with inpulse input image
    thresh

    Returns
    -------
    list of indices of bounding box of receptive filed
    """
    if thresh is None:
        thresh = 0
    return [[numpy.amin(i), numpy.amax(i)] for i in numpy.where(x != thresh)]


# Derive receptive field from the model
def receptive_field_conv(kernel, stride, n0=1, n_lyrs=1):
    """
    Compute receptive field for convolution layers

    Parameters
    ----------
    kernel
        kernel size
    stride
        stride size
    n0
        receptive field from previous layer
    n_lyrs
        number of layers

    Returns
    -------
    receptive field
    """
    n1 = n0
    for i in range(n_lyrs):
        n1 = kernel + (n1 - 1) * stride
    return n1


def receptive_field_pool(kernel, n0=1, shift_n=0, n_lyrs=1):
    """
    Compute receptive field for pooling layers

    Parameters
    ----------
    kernel
        kernel size
    n0
        receptive field from previous layer
    shift_n
        number of shifting pixels in shift convolution architecture
    n_lyrs
        number of layers

    Returns
    -------
    a tuple of two n1, s
    """
    n1 = n0 * kernel ** n_lyrs
    if shift_n > 1:  # shift_n is the Nst pooling lyr
        s = 1 + 2 ** shift_n
    else:
        s = shift_n
    return n1, s


def receptive_field_up(pl_size, n0=1):
    """
    Compute receptive field for up sampling layers

    Parameters
    ----------
    pl_size
        pooling size
    n0
        receptive field from previous layer

    Returns
    -------
    receptive field
    """
    return n0 if n0 == 1 else numpy.ceil(n0 * 1 / pl_size).astype(int)


def receptive_field_model(model, verbose=False):
    """
    Compute theoretical receptive field. Effective receptive field could be a square root of it.

    Parameters
    ----------
    model
        CNN model
    verbose

    Returns
    -------
    theoretical receptive field
    """
    n1 = 1  # starting field size to calculate receptive field
    shift_n = 0  # index of pooling lyr to calc. shift in shiftconv
    s = 0  # shift due to shiftconv
    shift = False
    layers = numpy.copy(model.layers)
    layers = list(layers)
    layers.reverse()
    for layer in layers:
        lyr_type = layer.__class__.__name__
        if 'Cropping' in lyr_type:
            shift = True
        elif 'Conv' in lyr_type:
            n1 = receptive_field_conv(layer.kernel_size[0], layer.strides[0], n1)
        elif 'Pooling' in lyr_type:
            if shift:
                shift_n += 1
            n1, s = receptive_field_pool(layer.pool_size[0], n1, shift_n)
        elif 'UpSampling' in lyr_type:
            n1 = receptive_field_up(layer.size[0], n1)
        if verbose:
            print(f'{lyr_type}: RF {n1}, shift {s}')
    if shift:
        n1 = (n1 + s) * 2
        print(
            f'Synthetic receptive field is {n1}^2 (with {int(n1/2 + s)}^2 empty space at 4 corners).'
        )
    return n1
