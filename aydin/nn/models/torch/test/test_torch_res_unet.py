import numpy
import pytest
import torch

from aydin.io.datasets import camera, normalise, add_noise
from aydin.nn.models.torch.torch_res_unet import ResidualUNetModel
from aydin.nn.models.torch.torch_unet import n2t_unet_train_loop
from aydin.nn.pytorch.it_ptcnn import to_numpy


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


def test_supervised_2D_n2t():
    visualize = False
    lizard_image = normalise(camera()[:256, :256])
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)
    lizard_image = torch.tensor(lizard_image)

    model = ResidualUNetModel(nb_unet_levels=2, supervised=True, spacetime_ndim=2)

    n2t_unet_train_loop(input_image, lizard_image, model)

    denoised = model(input_image)

    if visualize:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(to_numpy(lizard_image), name="groundtruth")
        viewer.add_image(to_numpy(input_image), name="noisy")
        viewer.add_image(to_numpy(denoised), name="denoised")

        napari.run()

    # assert result.shape == input_image.shape
    # assert result.dtype == input_image.dtype
