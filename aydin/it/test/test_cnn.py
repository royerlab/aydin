import time

import numpy
import pytest
import tensorflow as tf  # noqa: F401
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.python.keras.backend import clear_session

from aydin.io import io
from aydin.io.datasets import normalise, add_noise, examples_single
from aydin.it.cnn import ImageTranslatorCNN


def test_it_cnn_history():
    """
    Check if training history is properly recorded.
    """
    start = time.time()
    max_epochs = 2
    data = numpy.zeros((64, 64))
    it = ImageTranslatorCNN(
        model_architecture="unet",
        training_architecture='checkran',
        nb_unet_levels=1,
        patch_size=64,
        batch_size=1,
        mask_size=3,
        total_num_patches=1,
        patience=1,
        max_epochs=max_epochs,
    )
    it.train(data, data)
    history = it.loss_history
    for key, val in history.history.items():
        assert len(val) == max_epochs
    assert len(history.epoch) == max_epochs

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    clear_session()


def test_it_cnn_shiftconv_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 30
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    print("noisy shape: ", noisy.shape)

    it = ImageTranslatorCNN(
        model_architecture="unet",
        training_architecture='shiftconv',
        nb_unet_levels=2,
        batch_norm=None,  # 'instance',
        max_epochs=max_epochs,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    clear_session()


def test_it_cnn_checkerbox_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 5
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture="unet",
        training_architecture='checkerbox',
        nb_unet_levels=2,
        mask_size=3,
        batch_norm='instance',
        max_epochs=max_epochs,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    clear_session()


def test_it_cnn_random_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 5
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture="unet",
        training_architecture='random',
        nb_unet_levels=2,
        batch_norm='instance',
        max_epochs=max_epochs,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy * 0.9 and ssim_denoised > ssim_noisy * 0.9
    clear_session()


def test_it_cnn_checkran_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 5
    image_width = 100

    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    # Test with arbitrary input shape
    arbitrary_shape = (1, 1) + image.shape
    batch_dims = tuple([True if i == 1 else False for i in arbitrary_shape])
    image = image.reshape(arbitrary_shape)
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture="unet",
        training_architecture='checkran',
        nb_unet_levels=2,
        mask_size=3,
        batch_norm='instance',
        max_epochs=max_epochs,
    )
    it.train(noisy, noisy, batch_axes=batch_dims)
    denoised = it.translate(noisy, tile_size=image_width, batch_axes=batch_dims)
    assert denoised.shape == noisy.shape
    denoised = denoised.squeeze()
    noisy = noisy.squeeze()
    image = image.squeeze()

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    clear_session()


def test_it_cnn_jinet2D_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 30
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture='jinet', patch_size=image_width, max_epochs=max_epochs
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    clear_session()


def test_it_cnn_jinet2D_supervised_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 30
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture='jinet', patch_size=image_width, max_epochs=max_epochs
    )
    it.train(noisy, image)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    clear_session()


def test_it_cnn_jinet3D_light():
    """
    Demo for self-supervised denoising
    """
    start = time.time()
    max_epochs = 30
    image_width = 64
    image_path = examples_single.royerlab_hcr.get_path()
    image, metadata = io.imread(image_path)
    image = image[10:20, 1:2, 100 : 100 + image_width, 200 : 200 + image_width]
    image = rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture='jinet', patch_size=image_width, max_epochs=max_epochs
    )
    it.train(
        noisy, noisy, batch_axes=metadata.batch_axes, channel_axes=metadata.channel_axes
    )
    denoised = it.translate(
        noisy,
        tile_size=image_width,
        batch_axes=metadata.batch_axes,
        channel_axes=metadata.channel_axes,
    )

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    noisy = numpy.squeeze(noisy)
    image = numpy.squeeze(image)
    denoised = numpy.squeeze(denoised)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > (psnr_noisy * 0.5) and ssim_denoised > (ssim_noisy * 0.5)
    clear_session()


def test_it_cnn_jinet3D_supervised_light():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 30
    image_width = 64
    image_path = examples_single.royerlab_hcr.get_path()
    image, metadata = io.imread(image_path)
    image = image[10:20, 1:2, 100 : 100 + image_width, 200 : 200 + image_width]
    image = rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        model_architecture='jinet', patch_size=image_width, max_epochs=max_epochs
    )
    it.train(
        noisy, image, batch_axes=metadata.batch_axes, channel_axes=metadata.channel_axes
    )
    denoised = it.translate(
        noisy,
        tile_size=image_width,
        batch_axes=metadata.batch_axes,
        channel_axes=metadata.channel_axes,
    )

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    noisy = numpy.squeeze(noisy)
    image = numpy.squeeze(image)
    denoised = numpy.squeeze(denoised)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image, multichannel=True)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image, multichannel=True)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > (psnr_noisy * 0.5) and ssim_denoised > (ssim_noisy * 0.5)
    clear_session()


@pytest.mark.heavy
def test_it_cnn_random_patching():
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 16
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNN(
        training_architecture='random',
        nb_unet_levels=2,
        batch_norm='instance',
        max_epochs=max_epochs,
        patch_size=64,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop - start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
