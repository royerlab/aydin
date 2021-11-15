# flake8: noqa
# flake8: noqa

import numpy
from aydin.it.pytorch.lucyrichardson.clipped_lr import richardson_lucy_shrink
from scipy import ndimage
from scipy.fft import dct
from scipy.signal import convolve2d
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.resolution_estimate import signal_to_noise_ratio, spectrum_2d
from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import normalise, add_noise, newyork
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


deconv = richardson_lucy_shrink

image = newyork()
image = normalise(image.astype(numpy.float32))

psf = SimpleMicroscopePSF()
psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
print(psf_xyz_array.shape)
kernel_psf = psf_xyz_array[0]

blurred_image = convolve2d(image, kernel_psf, mode='same', boundary='symm')
noisy_and_blurred_image = add_noise(blurred_image, intensity=100, variance=0.001)

# noisy = numpy.stack([add_noise(blurred_image, intensity=1000, variance=0.01*i) for i in range(30)])
# noisy_spectrum = numpy.stack([spectrum(i, log=True)[0] for i in noisy])
# snr_values = [signal_to_noise_ratio(i) for i in noisy]
# print(snr_values)
#
# import napari
#
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(noisy, name='noisy')
#     viewer.add_image(noisy_spectrum, name='noisy_spectrum')


it = ImageTranslatorClassic(
    feature_generator=FastMultiscaleConvolutionalFeatures(exclude_scale_one=False),
    regressor=GBMRegressor(n_estimators=2000),
    normaliser_type='identity',
)

# Try denoising first:
it.train(noisy_and_blurred_image)
noisy_and_blurred_image = it.translate(noisy_and_blurred_image)

deconvolved_image = numpy.stack(
    [
        deconv(noisy_and_blurred_image, kernel_psf, iterations=i)
        for i in range(0, 60, 30)
    ]
)

supervised_ssim = [ssim(i, image) for i in deconvolved_image]
print(supervised_ssim)
print(supervised_ssim.index(max(supervised_ssim)))

supervised_psnr = [psnr(i, image) for i in deconvolved_image]
print(supervised_psnr)
print(supervised_psnr.index(max(supervised_psnr)))

input_image = noisy_and_blurred_image
mask = numpy.random.rand(*image.shape) < 0.001
masked_input_image = (
    input_image * ~mask
    + ndimage.median_filter(
        input_image, footprint=numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    )
    * mask
)

deconvolved_image = numpy.stack(
    [deconv(masked_input_image, kernel_psf, iterations=i) for i in range(0, 60, 6)]
)

reconvolved_image = numpy.stack(
    [convolve2d(i, kernel_psf, mode='same', boundary='symm') for i in deconvolved_image]
)

diff_image = numpy.stack([abs(i - input_image) for i in reconvolved_image])

deconvolved_spectrum = numpy.stack(
    [spectrum_2d(i, log=True)[0] for i in deconvolved_image]
)

similarity_to_input = list([ssim(input_image, i) for i in deconvolved_image])
ssim_indicator = list(
    [
        0.5 * ssim(input_image, d) + 0.5 * ssim(input_image, r)
        for d, r in zip(deconvolved_image, reconvolved_image)
    ]
)

print(similarity_to_input)
print(similarity_to_input.index(max(similarity_to_input)))
print(ssim_indicator)
print(ssim_indicator.index(max(ssim_indicator)))

mutual_info_deconv = list([mutual_info(input_image, i) for i in deconvolved_image])
mutual_info_reconv = list([mutual_info(input_image, i) for i in reconvolved_image])
mutual_info_indicator = list(
    [0.5 * i + 0.5 * j for i, j in zip(mutual_info_deconv, mutual_info_reconv)]
)

snr = list([signal_to_noise_ratio(i) for i in deconvolved_image])

from matplotlib import pyplot as plt

plt.plot(ssim_indicator[1:], label='ssim_indicator')
# plt.plot(mutual_info_indicator[1:], label='mutual_info_indicator')
plt.plot(similarity_to_input[1:], label='similarity')
# plt.plot(mutual_info_deconv[1:], label='mutual_info_deconv')
# plt.plot(mutual_info_reconv[1:], label='mutual_info_reconv')
plt.plot(snr[1:], label='snr')
plt.legend()
plt.show()

import napari

with napari.gui_qt():
    viewer = napari.Viewer()
    # viewer.add_image(module_deconvolved_image, name='module_deconvolved_image')
    # viewer.add_image(si_deconvolved_denoised_image, name='si_deconvolved_denoised_image')
    viewer.add_image(diff_image, name='diff_image')
    viewer.add_image(masked_input_image, name='masked_input_image')
    viewer.add_image(reconvolved_image, name='reconvolved_image')
    viewer.add_image(deconvolved_image, name='deconvolved_image')
    viewer.add_image(deconvolved_spectrum, name='deconvolved_spectrum')
    viewer.add_image(noisy_and_blurred_image, name='noisy_and_blurred_image')
    viewer.add_image(image, name='image')
