# flake8: noqa
import time

import numpy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, add_noise, add_blur_2d, dots
from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv
from aydin.nn.pytorch.it_ptcnn import PTCNNImageTranslator
from aydin.nn.pytorch.it_ptcnn_deconv import PTCNNDeconvolution


def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def demo(image, max_epochs=1500):
    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=1000, variance=0.001, sap=0.0001
    )

    it_denoise = PTCNNImageTranslator(
        max_epochs=max_epochs,
        patience=512,
        learning_rate=0.01,
        normaliser_type='identity',
    )

    start = time.time()
    it_denoise.train(noisy_and_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised_image = it_denoise.translate(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    it_deconv = PTCNNDeconvolution(
        max_epochs=max_epochs,
        patience=512,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
    )

    start = time.time()
    it_deconv.train(denoised_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    aydin_denoised_deconvolved_image = it_deconv.translate(denoised_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    lr = ImageTranslatorLRDeconv(
        psf_kernel=psf_kernel, max_num_iterations=10, backend="cupy"
    )
    lr.train(denoised_image)
    aydin_denoised_lr_deconvolved = lr.translate(denoised_image)

    lr.train(noisy_and_blurred_image)
    lr_deconvolved_image = lr.translate(noisy_and_blurred_image)

    # image = numpy.clip(image, 0, 1)
    # lr_deconvolved_image = numpy.clip(lr_deconvolved_image, 0, 1)
    # denoised_image = numpy.clip(denoised_image, 0, 1)
    # noisy_and_blurred_image = numpy.clip(noisy_and_blurred_image, 0, 1)
    # aydin_denoised_and_deconvolved_image = numpy.clip(
    #     aydin_denoised_and_deconvolved_image, 0, 1
    # )
    # aydin_denoised_lr_deconvolved = numpy.clip(
    #     aydin_denoised_lr_deconvolved, 0, 1
    # )

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
    printscore(
        "n&b image",
        psnr(image, noisy_and_blurred_image),
        spectral_mutual_information(image, noisy_and_blurred_image),
        mutual_information(image, noisy_and_blurred_image),
        ssim(image, noisy_and_blurred_image),
    )
    printscore(
        "den image",
        psnr(image, denoised_image),
        spectral_mutual_information(image, denoised_image),
        mutual_information(image, denoised_image),
        ssim(image, denoised_image),
    )
    printscore(
        "aydin:   ",
        psnr(image, aydin_denoised_deconvolved_image),
        spectral_mutual_information(image, aydin_denoised_deconvolved_image),
        mutual_information(image, aydin_denoised_deconvolved_image),
        ssim(image, aydin_denoised_deconvolved_image),
    )
    printscore(
        "aydin+lr ",
        psnr(image, aydin_denoised_lr_deconvolved),
        spectral_mutual_information(image, aydin_denoised_lr_deconvolved),
        mutual_information(image, aydin_denoised_lr_deconvolved),
        ssim(image, aydin_denoised_lr_deconvolved),
    )
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
        viewer.add_image(denoised_image, name='denoised_image')
        viewer.add_image(
            aydin_denoised_lr_deconvolved, name='aydin_denoised_lr_deconvolved'
        )
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
        viewer.add_image(
            aydin_denoised_deconvolved_image,
            name='aydin_denoised_and_deconvolved_image',
        )


image = dots()

demo(image)
