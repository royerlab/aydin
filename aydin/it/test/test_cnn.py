import numpy

from aydin.analysis.image_metrics import calculate_print_psnr_ssim
from aydin.io.datasets import camera, add_noise, normalise
from aydin.it.cnn_torch import ImageTranslatorCNNTorch


def test_it_cnn_jinet2D_light():
    train_and_evaluate_cnn(camera())


def train_and_evaluate_cnn(input_image):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    # max_epochs = 30
    image_width = 100
    image = normalise(input_image)
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model='jinet')
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised = calculate_print_psnr_ssim(
        image, noisy, denoised
    )

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
