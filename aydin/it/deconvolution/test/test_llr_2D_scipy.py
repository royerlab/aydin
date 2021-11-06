# flake8: noqa

import numpy
import pytest
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise, add_blur_2d, cropped_newyork
from aydin.it.deconvolution.llr_deconv import ImageTranslatorLearnedLRDeconv
from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.util.log.log import Log


@pytest.mark.heavy
def test():
    image = cropped_newyork()

    Log.enable_output = True

    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=20, variance=0.005, sap=0.001
    )

    lr = ImageTranslatorLearnedLRDeconv(
        psf_kernel=psf_kernel,
        max_num_iterations=30,
        deconvolution_class=ImageTranslatorLRDeconvScipy,
    )

    lr.train(noisy_and_blurred_image, noisy_and_blurred_image)
    lr_deconvolved_image = lr.deconvolve(noisy_and_blurred_image)
    llr_deconvolved_image = lr.translate(noisy_and_blurred_image)

    assert ssim(image, noisy_and_blurred_image) < 0.25
    assert ssim(image, lr_deconvolved_image) > 0.20
    assert ssim(image, llr_deconvolved_image) > 0.44


# image = newyork()
#
# demo(image)
