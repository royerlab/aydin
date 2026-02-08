"""Utility Keras layers for tensor manipulation (deprecated).

Provides Split, Swish activation, and Rot90 rotation layers used
in shift-convolution architectures.
"""

import keras.ops as ops
import tensorflow as tf
from deprecated import deprecated
from keras.layers import Layer


@deprecated(
    "All the Tensorflow related code and dependencies are deprecated and will be removed by v0.1.16"
)
class SplitLayer(Layer):
    """
    Split tensor at the batch axis. Only for shift convolution architecture.

    Parameters
    ----------
    idx : int
        index for the split chunk
    batchsize : int
        batch size
    """

    def __init__(self, idx, batchsize=1, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        self.batchsize = batchsize

    def call(self, x):
        return x[self.idx * self.batchsize : (self.idx + 1) * self.batchsize]

    def get_config(self):
        config = super().get_config()
        config.update({'idx': self.idx, 'batchsize': self.batchsize})
        return config


def Split(x, idx, batchsize=1, lyrname=None):
    """
    Split tensor at the batch axis. Only for shift convolution architecture.

    Parameters
    ----------
    x
        input tensor (unused, kept for backward compatibility)
    idx
        index for the split chunk
    batchsize
        batch size
    lyrname : string
        layer name

    Returns
    -------
    Split layer : SplitLayer

    """
    return SplitLayer(idx=idx, batchsize=batchsize, name=lyrname)


@deprecated(
    "All the Tensorflow related code and dependencies are deprecated and will be removed by v0.1.16"
)
class SwishLayer(Layer):
    """Swish activation layer."""

    def call(self, x):
        return tf.nn.swish(x)


def Swish(name=None):
    """
    Swish Layer

    Parameters
    ----------
    name : string

    Returns
    -------
    Swish layer : SwishLayer

    """
    return SwishLayer(name=name)


@deprecated(
    "All the Tensorflow related code and dependencies are deprecated and will be removed by v0.1.16"
)
class Rot90Layer(Layer):
    """
    Rotate tensor by 90 degrees for 2D, 3D images. Only for shift convolution architecture.

    Parameters
    ----------
    kk : int
        index for rotation (clockwise)
    """

    def __init__(self, kk=1, **kwargs):
        super().__init__(**kwargs)
        self.kk = kk

    def call(self, x):
        ndim = len(x.shape)

        if ndim == 4:
            tp_axis = [0, 2, 1, 3]  # (batch, longitudinal, horizontal, channel)
        elif ndim == 5:
            tp_axis = [0, 1, 3, 2, 4]  # (batch, z, longitudinal, horizontal, channel)
            tp_axis2 = [0, 3, 2, 1, 4]  # rotation along another axis
        else:
            raise ValueError(
                'Input shape has to be 4D or 5D. e.g. (Batch, (depth), longitudinal, horizontal, channel)'
            )

        kk = self.kk
        if kk < 0:
            direction = [-2, -3, -2, 1]
        else:
            direction = [-3, -2, 1, -2]

        # Convert negative axis to positive for tf.reverse
        def normalize_axis(axis, ndim):
            return axis if axis >= 0 else ndim + axis

        if abs(kk) % 6 == 5 and ndim == 5:
            axis = normalize_axis(direction[2], ndim)
            return tf.reverse(ops.transpose(x, tp_axis2), axis=[axis])
        elif kk % 6 == 0 and ndim == 5 and kk != 0:
            axis = normalize_axis(direction[3], ndim)
            return tf.reverse(ops.transpose(x, tp_axis2), axis=[axis])
        else:
            if kk % 4 == 1:
                axis = normalize_axis(direction[0], ndim)
                return tf.reverse(ops.transpose(x, tp_axis), axis=[axis])
            elif kk % 4 == 2:
                axis1 = normalize_axis(-2, ndim)
                axis2 = normalize_axis(-3, ndim)
                return tf.reverse(tf.reverse(x, axis=[axis1]), axis=[axis2])
            elif kk % 4 == 3:
                axis = normalize_axis(direction[1], ndim)
                return tf.reverse(ops.transpose(x, tp_axis), axis=[axis])
            elif kk % 4 == 0:
                return x

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        kk = self.kk
        if kk % 2 == 1 and 0 < kk % 6 < 5:
            # Swap the spatial dimensions for 90/270 degree rotations
            input_shape[-3], input_shape[-2] = input_shape[-2], input_shape[-3]
        elif abs(kk) % 6 == 5 or (kk % 6 == 0 and kk != 0):
            # For 3D rotations along another axis
            if len(input_shape) == 5:
                input_shape[1], input_shape[2] = input_shape[2], input_shape[1]
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'kk': self.kk})
        return config


def Rot90(xx, kk=1, lyrname=None):
    """
    Rotate tensor by 90 degrees for 2D, 3D images. Only for shift convolution architecture.

    Parameters
    ----------
    xx
        input tensor from previous layer (unused, kept for backward compatibility)
    kk
        index for rotation (clockwise)
    lyrname : string
        name of the layer

    Returns
    -------
    Rot90 layer : Rot90Layer
    """
    return Rot90Layer(kk=kk, name=lyrname)


class ReflectPad3D(Layer):
    """
    Asymmetric reflection padding for 3D tensors (internal use only).

    Parameters
    ----------
    padding : tuple of tuples
        Padding specification: ((d_before, d_after), (h_before, h_after), (w_before, w_after))
    """

    def __init__(self, padding=((0, 1), (0, 0), (0, 0)), **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, x):
        # For 5D tensor: (batch, depth, height, width, channels)
        # tf.pad expects padding for all dimensions
        pad_spec = [
            [0, 0],  # batch
            list(self.padding[0]),  # depth
            list(self.padding[1]),  # height
            list(self.padding[2]),  # width
            [0, 0],  # channels
        ]
        return tf.pad(x, pad_spec, mode='REFLECT')

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if input_shape[1] is not None:
            input_shape[1] += self.padding[0][0] + self.padding[0][1]
        if input_shape[2] is not None:
            input_shape[2] += self.padding[1][0] + self.padding[1][1]
        if input_shape[3] is not None:
            input_shape[3] += self.padding[2][0] + self.padding[2][1]
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config
