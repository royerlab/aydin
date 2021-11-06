from tensorflow.python.keras import Input, layers
from tensorflow.python.keras.layers import LeakyReLU, GaussianNoise, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1


def block(
    x,
    outputs=1,
    layers=1,
    layer_name=None,
    trainable=True,
    initialiser=None,
    weight_decay=0.0,
):

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
