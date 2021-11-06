from tensorflow.python.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    Activation,
    LeakyReLU,
    ZeroPadding3D,
    Conv3D,
    Cropping2D,
    Cropping3D,
    AveragePooling2D,
    MaxPooling2D,
    AveragePooling3D,
    MaxPooling3D,
    BatchNormalization,
)
from tensorflow.python.keras.regularizers import l1

from aydin.nn.layers.instance_norm import InstanceNormalization
from aydin.nn.layers.util import Swish


def conv2d_torch(
    x,
    out_channels,
    kernel_size,
    norm=None,
    act='ReLU',
    lyrname='cv2dtrch',
    padding=0,
    dilation_rate=1,
    bias=True,
    leaky_alpha=0.01,
):
    """
    Parameters
    ----------
    x
    out_channels
    kernel_size
    norm
    act
    lyrname
    padding
    dilation_rate
    bias
    leaky_alpha

    Returns
    -------
    conv2d_torch layer

    """
    x1 = ZeroPadding2D(padding, name=lyrname + '_pd0')(x)

    x1 = Conv2D(
        out_channels,
        kernel_size,
        dilation_rate=dilation_rate,
        name=lyrname + '_cv',
        use_bias=bias,
    )(x1)

    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act in ['lrel', 'leaky', 'LeakyReLU']:
        return LeakyReLU(alpha=leaky_alpha, name=lyrname + '_lrel')(x1)
    else:
        return x1


def conv3d_torch(
    x,
    out_channels,
    kernel_size,
    norm=None,
    act='ReLU',
    lyrname='cv3dtrch',
    padding=0,
    dilation_rate=1,
    bias=True,
    leaky_alpha=0.01,
):
    """

    Parameters
    ----------
    x
    out_channels
    kernel_size
    norm
    act
    lyrname
    padding
    dilation_rate
    bias
    leaky_alpha

    Returns
    -------
    conv3d_torch layer

    """
    x1 = ZeroPadding3D(padding, name=lyrname + '_pd0')(x)

    x1 = Conv3D(
        out_channels,
        kernel_size,
        dilation_rate=dilation_rate,
        name=lyrname + '_cv',
        use_bias=bias,
    )(x1)

    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act in ['lrel', 'leaky', 'LeakyReLU']:
        return LeakyReLU(alpha=leaky_alpha, name=lyrname + '_lrel')(x1)
    else:
        return x1


def conv2d_bn(
    xx,
    unit,
    kernel_size=3,
    shiftconv=False,
    norm=None,
    act='ReLU',
    weight_decay=0,
    lyrname=None,
):
    """

    Parameters
    ----------
    xx
    unit
    kernel_size
    shiftconv
    norm
    act
    weight_decay
    lyrname

    Returns
    -------
    conv2d_bn layer

    """
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(x1)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv2D(
            unit,
            kernel_size,
            padding='same',
            kernel_regularizer=l1(weight_decay),
            bias_regularizer=l1(weight_decay),
            name=lyrname + '_cv2',
        )(xx)

    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act == 'lrel':
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)
    else:
        return x1


def conv3d_bn(
    xx,
    unit,
    kernel_size=3,
    shiftconv=False,
    norm=None,
    act='ReLU',
    weight_decay=0,
    lyrname=None,
):
    """

    Parameters
    ----------
    xx
    unit
    kernel_size
    shiftconv
    norm
    act
    weight_decay
    lyrname

    Returns
    -------
    conv3d_bn layer

    """
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv3D(unit, (3, 3, 3), padding='same', name=lyrname + '_cv')(x1)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv3D(
            unit,
            kernel_size,
            padding='same',
            kernel_regularizer=l1(weight_decay),
            bias_regularizer=l1(weight_decay),
            name=lyrname + '_cv3',
        )(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act == 'lrel':
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)
    else:
        return x1


def pooling_down2D(xx, shiftconv, mode='max', lyrname=None):
    """

    Parameters
    ----------
    xx
    shiftconv
    mode
    lyrname

    Returns
    -------
    pooling_down2D layer

    """
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    if mode == 'ave':
        x1 = AveragePooling2D((2, 2), name=lyrname + '_apl')(x1)
    elif mode == 'max':
        x1 = MaxPooling2D((2, 2), name=lyrname + '_mpl')(x1)
    else:
        raise ValueError('pooling mode only accepts "max" or "ave".')
    return x1


def pooling_down3D(xx, shiftconv, pool_size=(2, 2, 2), mode='max', lyrname=None):
    """

    Parameters
    ----------
    xx
    shiftconv
    pool_size
    mode
    lyrname

    Returns
    -------
    pooling_down3D layer

    """
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    if mode == 'ave':
        x1 = AveragePooling3D(pool_size, name=lyrname + '_apl')(x1)
    elif mode == 'max':
        x1 = MaxPooling3D(pool_size, name=lyrname + '_mpl')(x1)
    else:
        raise ValueError('pooling mode only accepts "max" or "ave".')
    return x1
