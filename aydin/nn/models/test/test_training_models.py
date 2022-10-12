# flake8: noqa
import numpy
import torch

from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.res_unet import ResidualUNetModel
from aydin.nn.models.unet import UNetModel
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train


def test_supervised_2D_n2t():
    lizard_image = normalise(camera())
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)
    lizard_image = torch.tensor(lizard_image)

    model = ResidualUNetModel(
        nb_unet_levels=2,
        spacetime_ndim=2,
    )

    n2t_train(input_image, lizard_image, model, nb_epochs=2)
    result = model(input_image)

    assert result.shape == input_image.shape
    assert result.dtype == input_image.dtype


def test_supervised_2D_n2s():
    lizard_image = normalise(camera())
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)

    model = UNetModel(
        nb_unet_levels=2,
        spacetime_ndim=2,
    )

    n2s_train(input_image, model, nb_epochs=2)
    model.cpu()
    result = model(input_image)

    assert result.shape == input_image.shape
    assert result.dtype == input_image.dtype
