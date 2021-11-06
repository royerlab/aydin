# flake8: noqa
import time

import numpy
import pytest
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise, add_blur_2d, characters
from aydin.it.deconvolution.lr_deconv_tf import ImageTranslatorLRDeconvTensorflow
from aydin.util.log.log import lsection, lprint, Log


@pytest.mark.heavy
def test_lr_2d_tf():
    Log.enable_output = True

    image = characters()
    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=10000, variance=0.0001, sap=0.0000001
    )

    lr = ImageTranslatorLRDeconvTensorflow(psf_kernel=psf_kernel, max_num_iterations=20)

    with lsection("training:"):
        start = time.time()
        lr.train(noisy_and_blurred_image, noisy_and_blurred_image)
        stop = time.time()
        elapsed_train = stop - start

    with lsection("translating:"):
        start = time.time()
        lr_deconvolved_image = lr.translate(noisy_and_blurred_image)
        stop = time.time()
        elapsed_translate = stop - start

    lprint(f"elapsed_train={elapsed_train}")
    lprint(f"elapsed_translate={elapsed_translate}")

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(blurred_image, name='blurred')
    #     viewer.add_image(noisy_and_blurred_image, name='noisy')
    #     viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')

    assert elapsed_train < 1
    print("elapsed_translate: ", elapsed_translate)
    assert (
        elapsed_translate
        < 300  # something is wrong here, super slow.... not clear why...
    )  # already a long time with CPU for relatively small image

    assert ssim(image, noisy_and_blurred_image) < 0.90
    assert ssim(image, lr_deconvolved_image) > 0.81
