# flake8: noqa
import pytest
import torch

from aydin.nn.models.torch.torch_unet import UNetModel


def test_forward_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        spacetime_ndim=2,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5, 8])
def test_masking_2D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 1024, 1024))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=nb_unet_levels,
        spacetime_ndim=2,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        nb_unet_levels=2,
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5])
def test_masking_3D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=nb_unet_levels,
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# def test_various_masking_3D():
#     for i in [0, 4]:
#         input_array = torch.zeros((1, 21 + i, 64, 64, 1))
#         print(f'input shape: {input_array.shape}')
#         model3d = UNetModel(
#             input_array.shape[1:],
#             nb_unet_levels=4,
#             supervised=False,
#             spacetime_ndim=3,
#         )
#         result = model3d.predict([input_array, input_array])
#         assert result.shape == input_array.shape
#         assert result.dtype == input_array.dtype
#
#
# def test_thin_masking_3D():
#     for i in range(3):
#         input_array = torch.zeros((1, 2 + i, 64, 64, 1))
#         print(f'input shape: {input_array.shape}')
#         model3d = UNetModel(
#             input_array.shape[1:],
#             nb_unet_levels=4,
#             supervised=False,
#             spacetime_ndim=3,
#         )
#         result = model3d.predict([input_array, input_array])
#         assert result.shape == input_array.shape
#         assert result.dtype == input_array.dtype
