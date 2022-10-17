# flake8: noqa
import numpy

import torch

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.jinet import JINetModel
from aydin.nn.models.unet import UNetModel
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train


def test_2D_n2s_unet():
    run_2D(
        UNetModel(
            nb_unet_levels=2,
            spacetime_ndim=2,
        ),
        n2s_train,
        nb_epochs=128,
    )


def test_2D_n2s_jinet():
    run_2D(JINetModel(spacetime_ndim=2), n2s_train)


def test_2D_n2t_unet():
    run_2D(
        UNetModel(
            nb_unet_levels=2,
            spacetime_ndim=2,
        ),
        n2t_train,
    )


def test_2D_n2t_jinet():
    run_2D(JINetModel(spacetime_ndim=2), n2t_train)


def run_2D(model, train_method, nb_epochs=20):
    camera_image = normalise(camera())
    camera_image = numpy.expand_dims(camera_image, axis=0)
    camera_image = numpy.expand_dims(camera_image, axis=0)
    noisy_image = add_noise(camera_image)
    noisy_image = torch.tensor(noisy_image)

    if train_method == n2s_train:
        train_method(noisy_image, model, nb_epochs=nb_epochs)
    elif train_method == n2t_train:
        train_method(noisy_image, camera_image, model, nb_epochs=nb_epochs)

    model.cpu()
    denoised = model(noisy_image)

    camera_image = camera_image[0, 0, :, :]
    noisy_image = noisy_image.detach().numpy()[0, 0, :, :]
    denoised = denoised.detach().numpy()[0, 0, :, :]

    _, _, ssim_noisy, ssim_denoised = calculate_print_psnr_ssim(
        clean_image=camera_image, noisy_image=noisy_image, denoised_image=denoised
    )

    assert ssim_denoised > ssim_noisy
    assert ssim_denoised > 0.46
