# flake8: noqa
import time

import numpy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, add_noise, characters, add_blur_2d, newyork
from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.util.log.log import Log


def demo(image):
    Log.enable_output = True

    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=10000, variance=0.0001, sap=0.0000001
    )

    lr = ImageTranslatorLRDeconvScipy(psf_kernel=psf_kernel, max_num_iterations=200)

    start = time.time()
    lr.train(noisy_and_blurred_image, noisy_and_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    lr_deconvolved_image = lr.translate(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")

    def printscore(header, val1, val2, val3, val4):
        print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")

    printscore(
        "n&b image",
        psnr(image, noisy_and_blurred_image),
        spectral_mutual_information(image, noisy_and_blurred_image),
        mutual_information(image, noisy_and_blurred_image),
        ssim(image, noisy_and_blurred_image),
    )
    # printscore(
    #     "den image",
    #     psnr(image, denoised_image),
    #     spectral_mutual_information(image, denoised_image),
    #     mutual_information(image, denoised_image),
    #     ssim(image, denoised_image),
    # )
    printscore(
        "lr      ",
        psnr(image, lr_deconvolved_image),
        spectral_mutual_information(image, lr_deconvolved_image),
        mutual_information(image, lr_deconvolved_image),
        ssim(image, lr_deconvolved_image),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(noisy_and_blurred_image, name='noisy')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')


demo(newyork())
demo(characters())
