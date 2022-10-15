# flake8: noqa
import numpy
from scipy.fft import dct
from scipy.signal import convolve2d
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import richardson_lucy

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import add_noise, newyork
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def dctpsnr(true_image, test_image):
    true_image = n(dct(dct(true_image, axis=0), axis=1))
    test_image = n(dct(dct(test_image, axis=0), axis=1))

    return psnr(true_image, test_image)


# Prepare image:
image = newyork()
image = image.astype(numpy.float32)
image = n(image)

# Prepare PSF:
psf = SimpleMicroscopePSF()
psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
print(psf_xyz_array.shape)
kernel_psf = psf_xyz_array[0]

# degrade image with blurr and noise;
blurred_image = convolve2d(image, kernel_psf, 'same')
noisy_and_blurred_image = add_noise(
    blurred_image, intensity=10, variance=0.01, sap=0.00
)  # , intensity=100, variance=0.01, sap=0.001

# try classic_denoisers LR:
lr_deconvolved_image = richardson_lucy(
    noisy_and_blurred_image, kernel_psf, iterations=60
)

it = ImageTranslatorClassic(
    feature_generator=FastMultiscaleConvolutionalFeatures(exclude_scale_one=False),
    regressor=GBMRegressor(n_estimators=1000),
    normaliser_type='identity',
)

# lr deconvolution followed by denoising:
it.train(lr_deconvolved_image)
deconvolved_denoised_image = it.translate(lr_deconvolved_image)

# Try denoising and then lr deconvolution:
it.train(noisy_and_blurred_image)
denoised_image = it.translate(noisy_and_blurred_image)
denoised_deconvolved_image = richardson_lucy(denoised_image, kernel_psf, iterations=30)

# Train to translate input to denoised deconvolved, but without blind spot:
it = ImageTranslatorClassic(
    feature_generator=FastMultiscaleConvolutionalFeatures(exclude_scale_one=False),
    regressor=GBMRegressor(n_estimators=10),
    normaliser_type='identity',
)
it.train(noisy_and_blurred_image, denoised_deconvolved_image, force_jinv=False)
restored_image = it.translate(noisy_and_blurred_image)

# lr_reg_deconvolved_image = richardson_lucy_reg(noisy_and_blurred_image, kernel_psf, iterations=10)

# Clipping for comparison:
# lr_reg_deconvolved_image = numpy.clip(lr_reg_deconvolved_image, 0, 1)
lr_deconvolved_image = numpy.clip(lr_deconvolved_image, 0, 1)
denoised_image = numpy.clip(denoised_image, 0, 1)
deconvolved_denoised_image = numpy.clip(deconvolved_denoised_image, 0, 1)
denoised_deconvolved_image = numpy.clip(denoised_deconvolved_image, 0, 1)
restored_image = numpy.clip(restored_image, 0, 1)

# Compare results:
# print(
#     "lr_reg_deconvolved_image",
#     psnr(image, lr_reg_deconvolved_image),
#     dctpsnr(image, lr_reg_deconvolved_image),
#     ssim(image, lr_reg_deconvolved_image),
# )
print(
    "lr_deconvolved_image",
    psnr(image, lr_deconvolved_image),
    dctpsnr(image, lr_deconvolved_image),
    ssim(image, lr_deconvolved_image),
)
print(
    "denoised_image",
    psnr(image, denoised_image),
    dctpsnr(image, denoised_image),
    ssim(image, denoised_image),
)
print(
    "deconvolved_denoised_image",
    psnr(image, deconvolved_denoised_image),
    dctpsnr(image, deconvolved_denoised_image),
    ssim(image, deconvolved_denoised_image),
)
print(
    "denoised_deconvolved_image",
    psnr(image, denoised_deconvolved_image),
    dctpsnr(image, denoised_deconvolved_image),
    ssim(image, denoised_deconvolved_image),
)
print(
    "restored_image",
    psnr(image, restored_image),
    dctpsnr(image, restored_image),
    ssim(image, restored_image),
)

import napari

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(blurred_image, name='blurred_image')
    viewer.add_image(noisy_and_blurred_image, name='noisy_and_blurred_image')
    # viewer.add_image(lr_reg_deconvolved_image, name='lr_reg_deconvolved_image')
    viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
    viewer.add_image(denoised_image, name='denoised_image')
    viewer.add_image(deconvolved_denoised_image, name='deconvolved_denoised_image')
    viewer.add_image(denoised_deconvolved_image, name='denoised_deconvolved_image')
    viewer.add_image(restored_image, name='restored_image')
