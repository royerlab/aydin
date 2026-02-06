"""Neural network model architectures for the perceptron regressor.

Provides several feed-forward and residual network architectures built with
Keras, used as the underlying models in :class:`~aydin.regression.perceptron.PerceptronRegressor`.
"""

import keras
from keras import Input, layers
from keras.layers import Dense, GaussianNoise, LeakyReLU
from keras.models import Model
from keras.regularizers import l1


def block(
    x,
    outputs=1,
    layers=1,
    layer_name=None,
    trainable=True,
    initialiser=None,
    weight_decay=0.0,
):
    """Build a dense block consisting of one or more Dense + LeakyReLU layers.

    Parameters
    ----------
    x : keras.KerasTensor
        Input tensor.
    outputs : int
        Number of units in each dense layer.
    layers : int
        Number of dense layers in the block.
    layer_name : str
        Base name for layer naming.
    trainable : bool
        Whether the layers are trainable.
    initialiser : str or None
        Kernel initialiser name. Defaults to ``'glorot_uniform'``.
    weight_decay : float
        L1 regularisation weight for kernels and biases.

    Returns
    -------
    keras.KerasTensor
        Output tensor after the dense block.
    """
    for i in range(0, layers):

        x = Dense(
            outputs,
            name=layer_name + 'd1l' + str(i),
            trainable=trainable,
            use_bias=False,
            kernel_initializer='glorot_uniform' if initialiser is None else initialiser,
            bias_initializer='zeros',
            kernel_regularizer=l1(weight_decay),
            bias_regularizer=l1(weight_decay),
        )(x)
        # x = BatchNormalization(name=layer_name + 'bn1', center=True, scale=False)(x)
        x = LeakyReLU(name=layer_name + 'act1l' + str(i))(x)

    return x


def yinyang(feature_dim, depth=16):
    """Build a YinYang residual network with input concatenation shortcuts.

    Each block after the first receives the concatenation of the original
    input and the previous block's output. All intermediate outputs are
    summed before the final dense layer.

    Parameters
    ----------
    feature_dim : int
        Number of input features.
    depth : int
        Number of dense blocks.

    Returns
    -------
    keras.Model
        Compiled Keras model with a single scalar output.
    """
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature
    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        if d != 0:
            u = layers.concatenate([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model


def feed_forward_width(feature_dim, width=None, depth=16):
    """Build a feed-forward residual network with configurable width.

    If ``width`` exceeds ``feature_dim``, an initial dense layer expands
    the input. All intermediate outputs are summed before the final layer.

    Parameters
    ----------
    feature_dim : int
        Number of input features.
    width : int or None
        Width of each hidden layer. Defaults to ``feature_dim``.
    depth : int
        Number of dense blocks.

    Returns
    -------
    keras.Model
        Compiled Keras model with a single scalar output.
    """
    if not width:
        width = feature_dim

    initialiser = 'glorot_uniform'  # RandomNormal(stddev = 0.1)

    input = Input(shape=(feature_dim,), name='input')

    if width <= feature_dim:
        x = input
    else:
        x = Dense(
            width - feature_dim,
            name='fc_first',
            kernel_initializer=initialiser,
            trainable=True,
        )(input)
        x = layers.concatenate([input, x])

    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        x = block(x, outputs=width, layer_name=f'fc{d}', initialiser=initialiser)
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input, x)

    return model


def feed_forward(feature_dim, weight_decay=0.0001, depth=16, noise=None):
    """Build a feed-forward residual network with optional Gaussian noise.

    This is the default architecture used by :class:`PerceptronRegressor`.
    All intermediate outputs are summed (residual connections) before the
    final dense layer.

    Parameters
    ----------
    feature_dim : int
        Number of input features.
    weight_decay : float
        L1 regularisation weight applied to all dense layers.
    depth : int
        Number of dense blocks.
    noise : float or None
        Standard deviation of Gaussian noise added to the input. If
        ``None``, no noise is added.

    Returns
    -------
    keras.Model
        Compiled Keras model with a single scalar output.
    """
    width = feature_dim

    input = Input(shape=(feature_dim,), name='input')

    if noise is None:
        x = input
    else:
        x = GaussianNoise(noise)(input)

    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        x = block(x, outputs=width, layer_name=f'fc{d}', weight_decay=weight_decay)
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input, x)

    return model


def yinyang2(feature_dim, depth=8):
    """Build a YinYang v2 residual network with additive input shortcuts.

    Similar to :func:`yinyang` but uses element-wise addition (instead of
    concatenation) to combine the original input with the previous block's
    output.

    Parameters
    ----------
    feature_dim : int
        Number of input features.
    depth : int
        Number of dense blocks.

    Returns
    -------
    keras.Model
        Compiled Keras model with a single scalar output.
    """
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature
    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        if d != 0:
            u = layers.add([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model


def back_feed(feature_dim, depth=8):
    """Build a back-feed network with additive input shortcuts.

    Similar to :func:`yinyang2` but without the final residual sum of all
    intermediate outputs.

    Parameters
    ----------
    feature_dim : int
        Number of input features.
    depth : int
        Number of dense blocks.

    Returns
    -------
    keras.Model
        Compiled Keras model with a single scalar output.
    """
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature

    for d in range(0, depth):
        if d != 0:
            u = layers.add([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model
