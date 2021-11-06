import numpy


# Test masking method ====================================================================
from aydin.nn.models.unet import UNetModel
from aydin.nn.util.receptive_field import bbox_idx, receptive_field_model


def test_receptive_field_masking():
    batch_norm = None
    supervised = False
    shiftconv = False  # True
    batch_size = 1
    num_lyrs = 2
    input_image = numpy.zeros((1, 512, 512, 1))
    input_shape = input_image.shape
    input_image[0, int(input_shape[1] / 2), int(input_shape[2] / 2), 0] = 1
    model = UNetModel(
        input_shape[1:],
        mini_batch_size=1,
        nb_unet_levels=num_lyrs,
        normalization=batch_norm,
        supervised=supervised,
        shiftconv=shiftconv,
        spacetime_ndim=2,
    )
    # Predict test 0-image
    if not shiftconv and not supervised:
        pred1 = model.predict(
            [input_image, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
        )
    else:
        pred1 = model.predict(input_image, batch_size=batch_size, verbose=1)

    # Calculate receptive field
    receptive_field_idx = numpy.array(bbox_idx(pred1))
    receptive_field_size = receptive_field_idx[:, -1] - receptive_field_idx[:, 0] + 1

    rf_size_test = receptive_field_model(model, verbose=True)
    assert rf_size_test == receptive_field_size[1:2]


# # Test shiftconv method ====================================================================
# def test_receptive_field_shiftconv():
#     batch_norm = None
#     supervised = False
#     shiftconv = True  # True
#     batch_size = 1
#     num_lyrs = 5
#     input_image = numpy.zeros((1, 512, 512, 1))
#     input_shape = input_image.shape
#     input_image[0, int(input_shape[1] / 2), int(input_shape[2] / 2), 0] = 1
#     model = UNet2DModel(
#         input_shape[1:],
#         rot_batch_size=1,
#         num_lyr=num_lyrs,
#         normalization=batch_norm,
#         supervised=supervised,
#         shiftconv=shiftconv,
#     )
#     # Predict test 0-image
#     if not shiftconv and not supervised:
#         pred1 = model.predict(
#             [input_image, numpy.ones(input_shape)], batch_size=batch_size, verbose=1
#         )
#     else:
#         pred1 = model.predict(input_image, batch_size=batch_size, verbose=1)
#
#     # Calculate receptive field
#     receptive_field_idx = numpy.array(bbox_idx(pred1))
#     receptive_field_size = receptive_field_idx[:, -1] - receptive_field_idx[:, 0] + 1
#
#     rf_size_test = receptive_field_model(model, verbose=True)
#     assert rf_size_test == receptive_field_size[1:2]
