import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Lambda


def Split(
    x, idx, batchsize=1, lyrname=None
):  # TODO: refactor into a class or a util function
    """
    Split tensor at the batch axis. Only for shift convolution architecture.

    Parameters
    ----------
    x
        input tensor
    idx
        index for the split chunk
    batchsize
        batch size
    lyrname : string
        layer name

    Returns
    -------
    Split layer : layers.Lambda

    """
    out_shape = backend.int_shape(x[0])
    return Lambda(
        lambda xx: xx[idx * batchsize : (idx + 1) * batchsize],
        output_shape=out_shape,
        name=lyrname,
    )


def Swish(name=None):
    """
    Swish Layer

    Parameters
    ----------
    name : string

    Returns
    -------
    Swish layer : layers.Lambda

    """
    return Lambda(tf.nn.swish, name=name)


def Rot90(xx, kk=1, lyrname=None):
    """
    Rotate tensor by 90 degrees for 2D, 3D images. Only for shift convolution architecture.

    Parameters
    ----------
    xx
        input tensor from previous layer
    kk
        index for rotation (crock wise)
    lyrname : string
        name of the layer

    Returns
    -------
    Rot90 layer : layers.Lambda
    """
    out_shape = list(backend.int_shape(xx))
    if kk % 2 == 1 and 0 < kk % 6 < 5:
        out_shape[-3:-1] = np.flip(out_shape[-3:-1], 0)
    elif abs(kk) % 6 == 5 or kk % 6 == 0:
        out_shape[1:3] = np.flip(out_shape[1:3], 0)
    if len(out_shape) == 4:
        tp_axis = [0, 2, 1, 3]  # (batch, longitudinal, horizontal, channel)
    elif len(out_shape) == 5:
        tp_axis = [
            0,
            1,
            3,
            2,
            4,
        ]  # (batch, z-direction, longitudinal, horizontal, channel)
        tp_axis2 = (0, 3, 2, 1, 4)  # rotation along another axis
    else:
        raise ValueError(
            'Input shape has to be 4D or 5D. e.g. (Batch, (depth), longitudinal, horizontal, channel)'
        )

    if kk < 0:
        direction = [-2, -3, -2, 1]
    else:
        direction = [-3, -2, 1, -2]
    if abs(kk) % 6 == 5 and len(out_shape) == 5:
        return Lambda(
            lambda xx: backend.reverse(
                backend.permute_dimensions(xx, tp_axis2), axes=direction[2]
            ),
            output_shape=out_shape[1:],
            name=lyrname,
        )
    elif kk % 6 == 0 and len(out_shape) == 5 and kk != 0:
        return Lambda(
            lambda xx: backend.reverse(
                backend.permute_dimensions(xx, tp_axis2), axes=direction[3]
            ),
            output_shape=out_shape[1:],
            name=lyrname,
        )
    else:
        if kk % 4 == 1:
            return Lambda(
                lambda xx: backend.reverse(
                    backend.permute_dimensions(xx, tp_axis), axes=direction[0]
                ),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 2:
            return Lambda(
                lambda xx: backend.reverse(backend.reverse(xx, axes=-2), axes=-3),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 3:
            return Lambda(
                lambda xx: backend.reverse(
                    backend.permute_dimensions(xx, tp_axis), axes=direction[1]
                ),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 0:
            return Lambda(lambda xx: xx, output_shape=out_shape[1:], name=lyrname)
