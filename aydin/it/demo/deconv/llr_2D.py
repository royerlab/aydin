# flake8: noqa
import time

import numpy
from aydin.features.fast.fast_features import FastFeatureGenerator
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, add_noise, add_blur_2d, characters
from aydin.it.deconvolution.llr_deconv import ImageTranslatorLearnedLRDeconv
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image):
    Log.enable_output = True

    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=20, variance=0.005, sap=0.001
    )

    lr = ImageTranslatorLearnedLRDeconv(psf_kernel=psf_kernel, max_num_iterations=30)

    start = time.time()
    lr.train(noisy_and_blurred_image, noisy_and_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    lr_deconvolved_image = lr.deconvolve(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    start = time.time()
    llr_deconvolved_image = lr.translate(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    generator = FastFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = CBRegressor(patience=128)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(lr_deconvolved_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised_lr_deconvolved_image = it.translate(lr_deconvolved_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")

    def printscore(header, val1, val2, val3, val4):
        print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")

    printscore(
        "n&b image ",
        psnr(image, noisy_and_blurred_image),
        spectral_mutual_information(image, noisy_and_blurred_image),
        mutual_information(image, noisy_and_blurred_image),
        ssim(image, noisy_and_blurred_image),
    )
    printscore(
        "lr        ",
        psnr(image, lr_deconvolved_image),
        spectral_mutual_information(image, lr_deconvolved_image),
        mutual_information(image, lr_deconvolved_image),
        ssim(image, lr_deconvolved_image),
    )
    printscore(
        "denoised lr",
        psnr(image, denoised_lr_deconvolved_image),
        spectral_mutual_information(image, denoised_lr_deconvolved_image),
        mutual_information(image, denoised_lr_deconvolved_image),
        ssim(image, denoised_lr_deconvolved_image),
    )
    printscore(
        "learned lr",
        psnr(image, llr_deconvolved_image),
        spectral_mutual_information(image, llr_deconvolved_image),
        mutual_information(image, llr_deconvolved_image),
        ssim(image, llr_deconvolved_image),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(noisy_and_blurred_image, name='noisy')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
        viewer.add_image(
            denoised_lr_deconvolved_image, name='denoised_lr_deconvolved_image'
        )
        viewer.add_image(llr_deconvolved_image, name='llr_deconvolved_image')


image = characters()[0:1024, 0:1024]

demo(image)
