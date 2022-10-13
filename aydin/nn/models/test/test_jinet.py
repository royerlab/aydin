import numpy
import torch

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import add_noise, normalise, camera
from aydin.nn.models.jinet import JINetModel
from aydin.nn._legacy_pytorch.it_ptcnn import to_numpy
from aydin.nn.training_methods.n2t import n2t_train


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
