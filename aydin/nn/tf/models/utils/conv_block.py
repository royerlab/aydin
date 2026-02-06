"""Convolution block utilities for TensorFlow/Keras models (deprecated).

Provides 2D and 3D convolution blocks with optional normalization and
activation, as well as pooling downsampling functions. Supports both
standard and shift-convolution architectures.
"""

import keras
from deprecated import deprecated
from keras.layers import (
    Activation,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Cropping2D,
    Cropping3D,
    LeakyReLU,
    MaxPooling2D,
    MaxPooling3D,
    ZeroPadding2D,
    ZeroPadding3D,
)
from keras.regularizers import l1

from aydin.nn.tf.layers.instance_norm import InstanceNormalization
from aydin.nn.tf.layers.util import Swish


@deprecated(
    "All the Tensorflow related code and dependencies are deprecated and will be removed by v0.1.16"
)
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
    """Create a 2D convolution block with padding, normalization, and activation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    norm : str or None
        Normalization type: ``'instance'``, ``'batch'``, or ``None``.
    act : str
        Activation type: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
    lyrname : str
        Base name for layer naming.
    padding : int
        Amount of zero padding.
    dilation_rate : int
        Dilation rate for the convolution.
    bias : bool
        Whether to use bias in the convolution.
    leaky_alpha : float
        Negative slope for LeakyReLU activation.

    Returns
    -------
    tf.Tensor
        Output tensor after convolution, normalization, and activation.
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
    """Create a 3D convolution block with padding, normalization, and activation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    norm : str or None
        Normalization type: ``'instance'``, ``'batch'``, or ``None``.
    act : str
        Activation type: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
    lyrname : str
        Base name for layer naming.
    padding : int
        Amount of zero padding.
    dilation_rate : int
        Dilation rate for the convolution.
    bias : bool
        Whether to use bias in the convolution.
    leaky_alpha : float
        Negative slope for LeakyReLU activation.

    Returns
    -------
    tf.Tensor
        Output tensor after convolution, normalization, and activation.
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
    """Create a 2D convolution block for UNet with optional shift-convolution.

    Parameters
    ----------
    xx : tf.Tensor
        Input tensor.
    unit : int
        Number of output filters.
    kernel_size : int
        Convolution kernel size.
    shiftconv : bool
        If ``True``, applies padding and cropping for shift-convolution.
    norm : str or None
        Normalization type: ``'instance'``, ``'batch'``, or ``None``.
    act : str
        Activation type: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
    weight_decay : float
        L1 regularization coefficient.
    lyrname : str
        Base name for layer naming.

    Returns
    -------
    tf.Tensor
        Output tensor after convolution, normalization, and activation.
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
    """Create a 3D convolution block for UNet with optional shift-convolution.

    Parameters
    ----------
    xx : tf.Tensor
        Input tensor.
    unit : int
        Number of output filters.
    kernel_size : int
        Convolution kernel size.
    shiftconv : bool
        If ``True``, applies padding and cropping for shift-convolution.
    norm : str or None
        Normalization type: ``'instance'``, ``'batch'``, or ``None``.
    act : str
        Activation type: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
    weight_decay : float
        L1 regularization coefficient.
    lyrname : str
        Base name for layer naming.

    Returns
    -------
    tf.Tensor
        Output tensor after convolution, normalization, and activation.
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
    """Apply 2D spatial downsampling with optional shift-convolution padding.

    Parameters
    ----------
    xx : tf.Tensor
        Input tensor.
    shiftconv : bool
        If ``True``, applies padding and cropping for shift-convolution.
    mode : str
        Pooling mode: ``'max'`` or ``'ave'``.
    lyrname : str
        Base name for layer naming.

    Returns
    -------
    tf.Tensor
        Spatially downsampled tensor.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'max'`` or ``'ave'``.
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
    """Apply 3D spatial downsampling with optional shift-convolution padding.

    Parameters
    ----------
    xx : tf.Tensor
        Input tensor.
    shiftconv : bool
        If ``True``, applies padding and cropping for shift-convolution.
    pool_size : tuple of int
        Pooling window size for each spatial dimension.
    mode : str
        Pooling mode: ``'max'`` or ``'ave'``.
    lyrname : str
        Base name for layer naming.

    Returns
    -------
    tf.Tensor
        Spatially downsampled tensor.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'max'`` or ``'ave'``.
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
