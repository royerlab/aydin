import numpy

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import (
    add_noise,
    normalise,
    newyork,
    examples_single,
    camera,
)
from aydin.it.cnn_torch import ImageTranslatorCNNTorch


def test_it_cnn_jinet2D_light():
    train_and_evaluate_cnn(camera(), model="jinet")


def test_it_cnn_jinet3D_light():
    train_and_evaluate_cnn(
        examples_single.myers_tribolium.get_array()[:32, :32, :32], model="jinet"
    )


def test_it_cnn_unet2d():
    train_and_evaluate_cnn(camera(), model="unet")


def test_it_cnn_unet3d():
    train_and_evaluate_cnn(
        examples_single.janelia_flybrain.get_array()[:32, 1:2, :32, :32],
        model="unet",
    )


def train_and_evaluate_cnn(input_image, model="jinet"):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    # max_epochs = 30
    # image_width = 100
    image = normalise(input_image)
    # H0, W0 = (numpy.array(image.shape) - image_width) // 2
    # image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model=model)
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image.shape[0])

    image = numpy.squeeze(numpy.clip(image, 0, 1))
    noisy = numpy.squeeze(numpy.clip(noisy.reshape(image.shape), 0, 1))
    denoised = numpy.squeeze(numpy.clip(denoised, 0, 1))

    psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised = calculate_print_psnr_ssim(
        image, noisy, denoised
    )

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
