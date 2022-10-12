# flake8: noqa
import numpy

import torch

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import add_noise, camera, normalise
from aydin.nn.models.torch_jinet import JINetModel
from aydin.nn.models.torch_res_unet import ResidualUNetModel
from aydin.nn.models.torch_unet import UNetModel
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train


def test_supervised_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = ResidualUNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        spacetime_ndim=2,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


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


def test_2D_n2s_unet():
    run_2D_n2s(
        UNetModel(
            nb_unet_levels=2,
            spacetime_ndim=2,
        )
    )


def test_2D_n2s_jinet():
    run_2D_n2s(JINetModel(spacetime_ndim=2))


def run_2D_n2s(model):
    camera_image = normalise(camera())
    camera_image = numpy.expand_dims(camera_image, axis=0)
    camera_image = numpy.expand_dims(camera_image, axis=0)
    noisy_image = add_noise(camera_image)
    noisy_image = torch.tensor(noisy_image)

    n2s_train(noisy_image, model, nb_epochs=20)
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


def test_supervised_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=2,
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
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# def test_various_masking_3D():
#     for i in [0, 4]:
#         input_array = models.zeros((1, 21 + i, 64, 64, 1))
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
#         input_array = models.zeros((1, 2 + i, 64, 64, 1))
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
