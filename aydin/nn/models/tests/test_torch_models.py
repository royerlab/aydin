"""Integration tests for PyTorch model training and denoising quality.

Tests that each model architecture (UNet, JINet, ResUNet, LSUNet, DnCNN,
RonnebergerUNet) can be trained with N2S or N2T methods and produces
meaningful denoising quality.
"""

import numpy
import pytest
import torch

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.dncnn import DnCNNModel
from aydin.nn.models.jinet import JINetModel
from aydin.nn.models.linear_scaling_unet import LinearScalingUNetModel
from aydin.nn.models.res_unet import ResidualUNetModel
from aydin.nn.models.ronneberger_unet import RonnebergerUNetModel
from aydin.nn.models.unet import UNetModel
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train


@pytest.mark.parametrize(
    "model, train_method, nb_epochs",
    [
        (
            UNetModel(
                nb_unet_levels=2,
                spacetime_ndim=2,
            ),
            n2s_train,
            128,
        ),
        (JINetModel(spacetime_ndim=2), n2s_train, 40),
        (
            UNetModel(
                nb_unet_levels=2,
                spacetime_ndim=2,
            ),
            n2t_train,
            128,
        ),
        (JINetModel(spacetime_ndim=2), n2t_train, 40),
        (
            ResidualUNetModel(
                nb_unet_levels=2,
                spacetime_ndim=2,
            ),
            n2s_train,
            128,
        ),
        (
            LinearScalingUNetModel(
                nb_unet_levels=2,
                spacetime_ndim=2,
            ),
            n2s_train,
            128,
        ),
        (DnCNNModel(spacetime_ndim=2, num_of_layers=5), n2s_train, 40),
        (DnCNNModel(spacetime_ndim=2, num_of_layers=5), n2t_train, 40),
        (RonnebergerUNetModel(spacetime_ndim=2, depth=2), n2s_train, 128),
        (RonnebergerUNetModel(spacetime_ndim=2, depth=2), n2t_train, 128),
    ],
)
def test_models_2D(model, train_method, nb_epochs):
    """Test that 2D model training improves SSIM over noisy input."""
    numpy.random.seed(42)
    torch.manual_seed(42)
    camera_image = normalise(camera())
    camera_image = numpy.expand_dims(camera_image, axis=0)
    camera_image = numpy.expand_dims(camera_image, axis=0)
    noisy_image = add_noise(camera_image)
    noisy_image = torch.from_numpy(noisy_image)

    if train_method == n2s_train:
        train_method(noisy_image, model, nb_epochs=nb_epochs)
    elif train_method == n2t_train:
        train_method(noisy_image, camera_image, model, nb_epochs=nb_epochs)

    model.cpu()
    model.eval()
    with torch.no_grad():
        denoised = model(noisy_image)

    camera_image = camera_image[0, 0, :, :]
    noisy_image = noisy_image.detach().numpy()[0, 0, :, :]
    denoised = denoised.detach().numpy()[0, 0, :, :]

    _, _, ssim_noisy, ssim_denoised = calculate_print_psnr_ssim(
        clean_image=camera_image, noisy_image=noisy_image, denoised_image=denoised
    )

    assert ssim_denoised > ssim_noisy
