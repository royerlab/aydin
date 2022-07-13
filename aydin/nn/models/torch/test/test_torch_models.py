# flake8: noqa

# import math
# from collections import OrderedDict
# from itertools import chain
#
# import napari
import numpy
import pytest
import torch

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.torch.torch_unet import UNetModel, n2t_unet_train_loop
from aydin.nn.models.utils.torch_dataset import TorchDataset

# from aydin.nn.pytorch.it_ptcnn import to_numpy
# from aydin.nn.pytorch.optimizers.esadam import ESAdam
# from aydin.util.log.log import lprint


def test_supervised_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=2,
        residual=True,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_2D_n2t():
    lizard_image = normalise(camera()[:128, :128])
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)
    lizard_image = torch.tensor(lizard_image)

    # dataset = TorchDataset(input_image, lizard_image, 64, self_supervised=False)

    # data_loader = DataLoader(
    #     dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    # )

    model = UNetModel(
        nb_unet_levels=2, supervised=True, spacetime_ndim=2, residual=True
    )

    n2t_unet_train_loop(input_image, lizard_image, model)

    # assert result.shape == input_image.shape
    # assert result.dtype == input_image.dtype


def test_masking_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d(input_array, torch.ones(input_array.shape))
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# def test_jinet_2D():
#     input_array = torch.zeros((1, 1, 64, 64))
#     model2d = JINetModel((64, 64, 1), spacetime_ndim=2)
#     result = model2d.predict([input_array])
#     assert result.shape == input_array.shape
#     assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=2,
        supervised=False,
        spacetime_ndim=3,
    )
    result = model3d(input_array, torch.ones(input_array.shape))
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
