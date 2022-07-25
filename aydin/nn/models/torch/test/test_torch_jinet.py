import torch

from aydin.nn.models.torch.torch_jinet import JINetModel


def test_forward_2D_jinet():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = JINetModel(spacetime_ndim=2)
    result = model2d(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
