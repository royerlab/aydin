# flake8: noqa
import pytest
import torch

from aydin.nn.models.res_unet import ResidualUNetModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim, nb_unet_levels",
    [
        ((1, 1, 1024, 1024), 2, 2),
        ((1, 1, 1024, 1024), 2, 3),
        ((1, 1, 1024, 1024), 2, 5),
        ((1, 1, 1024, 1024), 2, 8),
        ((1, 1, 64, 64, 64), 3, 2),
        ((1, 1, 64, 64, 64), 3, 3),
        ((1, 1, 64, 64, 64), 3, 5),
    ],
)
def test_forward(input_array_shape, spacetime_ndim, nb_unet_levels):
    input_array = torch.zeros(input_array_shape)
    model = ResidualUNetModel(
        nb_unet_levels=nb_unet_levels,
        spacetime_ndim=spacetime_ndim,
    )
    result = model(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
