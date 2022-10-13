import torch

from aydin.nn.models.jinet import JINetModel


def test_forward_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = JINetModel(spacetime_ndim=2)
    result = model2d(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_forward_3D():
    input_array = torch.zeros((1, 1, 128, 128, 128))
    model3d = JINetModel(spacetime_ndim=3)
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
