# flake8: noqa
import numpy
import pytest
import torch

from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.torch.torch_unet import UNetModel, n2t_train
from aydin.nn.pytorch.it_ptcnn import to_numpy


def test_supervised_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=2,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# @pytest.mark.heavy
def test_supervised_2D_n2t():
    visualize = False
    lizard_image = normalise(camera()[:256, :256])
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    noisy_image = add_noise(lizard_image)

    noisy_image = torch.tensor(noisy_image)
    clean_image = torch.tensor(lizard_image)

    model = UNetModel(nb_unet_levels=2, supervised=True, spacetime_ndim=2)

    n2t_train(noisy_image, clean_image, model)

    denoised = model(noisy_image)

    if visualize:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(to_numpy(lizard_image), name="groundtruth")
        viewer.add_image(to_numpy(noisy_image), name="noisy")
        viewer.add_image(to_numpy(denoised), name="denoised")

        napari.run()

    assert denoised.shape == noisy_image.shape
    assert denoised.dtype == noisy_image.dtype


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5, 8])
def test_masking_2D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 1024, 1024))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=nb_unet_levels,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d(input_array, torch.ones(input_array.shape))
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


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


@pytest.mark.parametrize("nb_unet_levels", [2, 3, 5])
def test_masking_3D(nb_unet_levels):
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=nb_unet_levels,
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
