# flake8: noqa
import numpy
import matplotlib.pyplot as plt
from aydin.it.cnn.models.unet_2d import UNet2DModel


def bbox_idx(x, thresh=None):
    if thresh is None:
        thresh = 0
    return [[numpy.amin(i), numpy.amax(i)] for i in numpy.where(x != thresh)]


# Check the receptive field of masking approach ==========================================
def demo_masking():
    # Setup a model
    batch_norm = None
    supervised = False
    shiftconv = False  # True
    batch_size = 1
    num_lyr = 5
    input_image = numpy.zeros((1, 512, 512, 1))
    input_shape = input_image.shape
    input_image1 = numpy.copy(input_image)
    input_image1[0, int(input_shape[1] / 2), int(input_shape[2] / 2), 0] = 1
    model = UNet2DModel(
        input_shape[1:],
        rot_batch_size=1,
        num_lyr=num_lyr,
        normalization=batch_norm,
        supervised=supervised,
        shiftconv=shiftconv,
        res_node=1,
    )
    # run the model with a 0 image
    if not shiftconv and not supervised:
        pred0 = model.predict(
            [input_image, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
        )
        pred1 = model.predict(
            [input_image1, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
        )
    else:
        pred0 = model.predict(input_image, batch_size=batch_size, verbose=1)
        pred1 = model.predict(input_image1, batch_size=batch_size, verbose=1)

    # calculate receptive field
    receptive_field_idx = numpy.array(bbox_idx(pred1))
    receptive_field_size = receptive_field_idx[:, -1] - receptive_field_idx[:, 0] + 1
    print(f'rf size: {receptive_field_size}')

    # Visualize receptive field (by passing only 1 pixel info)
    pred01 = numpy.copy(pred1)
    pred01[pred01 == 0] = numpy.amin(pred1)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pred1.squeeze())
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred01.squeeze())
    plt.axis('off')
    plt.title('Masking approach')
    plt.show()


demo_masking()


# Check the receptive field of shiftconv approach ==========================================
def demo_shiftconv():
    batch_norm = None
    supervised = False
    shiftconv = True
    batch_size = 1
    num_lyr = 5
    input_image = numpy.zeros((1, 512, 512, 1))
    input_shape = input_image.shape
    input_image1 = numpy.copy(input_image)
    input_image1[0, int(input_shape[1] / 2), int(input_shape[2] / 2), 0] = 1
    model = UNet2DModel(
        input_shape[1:],
        rot_batch_size=1,
        num_lyr=num_lyr,
        normalization=batch_norm,
        supervised=supervised,
        shiftconv=shiftconv,
        res_node=1,
    )
    # run the model with a 0 image
    if not shiftconv and not supervised:
        pred0 = model.predict(
            [input_image, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
        )
        pred1 = model.predict(
            [input_image1, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
        )
    else:
        pred0 = model.predict(input_image, batch_size=batch_size, verbose=1)
        pred1 = model.predict(input_image1, batch_size=batch_size, verbose=1)

    # calculate receptive field
    receptive_field_idx = numpy.array(bbox_idx(pred1))
    receptive_field_size = receptive_field_idx[:, -1] - receptive_field_idx[:, 0] + 1
    print(f'rf size: {receptive_field_size}')

    # Visualize receptive field (by passing only 1 pixel info)
    pred01 = numpy.copy(pred1)
    pred01[pred01 == 0] = numpy.amin(pred1)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pred1.squeeze())
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred01.squeeze())
    plt.axis('off')
    plt.title('Shiftconv approach')
    plt.show()


demo_shiftconv()


# Derive receptive field from the model
def receptive_field_conv(kernel, stride, n0=1, n_lyrs=1):
    n1 = n0
    for i in range(n_lyrs):
        n1 = kernel + (n1 - 1) * stride
    return n1


def receptive_field_pool(kernel, n0=1, n_lyrs=1):
    n1 = n0 * kernel ** n_lyrs
    return n1


def receptive_field_pool(kernel, n0=1, shift_n=0, n_lyrs=1):
    n1 = n0 * kernel ** n_lyrs
    if shift_n > 1:  # shift_n is the Nst pooling lyr
        s = 1 + 2 ** shift_n
    else:
        s = shift_n
    return n1, s


def receptive_field_up(pl_size, n0=1):
    if n0 == 1:
        return n0
    else:
        return numpy.ceil(n0 * 1 / pl_size).astype(int)


def receptive_field_model(model, verbose=False):
    n1 = 1  # starting field size to calculate receptive field
    shift_n = 0  # index of pooling lyr to calc. shift in shiftconv
    s = 0  # shift due to shiftconv
    layers = numpy.copy(model.layers)
    layers = list(layers)
    layers.reverse()
    for layer in layers:
        lyr_type = layer.__class__.__name__
        if 'Padding' in lyr_type:
            shift = True
        if 'Conv' in lyr_type:
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
