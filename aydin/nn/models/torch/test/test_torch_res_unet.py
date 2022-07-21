import pytest
import torch

from aydin.nn.models.torch.torch_res_unet import ResidualUNetModel


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5, 8])
def test_masking_2D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 1024, 1024))
    model2d = ResidualUNetModel(
        nb_unet_levels=nb_unet_levels,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d(input_array, torch.ones(input_array.shape))
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5])
def test_masking_3D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = ResidualUNetModel(
        nb_unet_levels=nb_unet_levels,
        supervised=False,
        spacetime_ndim=3,
    )
    result = model3d(input_array, torch.ones(input_array.shape))
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
