import pytest
import torch

from aydin.nn.models.jinet import JINetModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim",
    [
        ((1, 1, 64, 64), 2),
        ((1, 1, 128, 128, 128), 3),
    ],
)
def test_forward(input_array_shape, spacetime_ndim):
    input_array = torch.zeros(input_array_shape)
    model = JINetModel(spacetime_ndim=spacetime_ndim)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
