# flake8: noqa
import time

import numpy

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, add_blur_2d, dots
from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv
from aydin.nn.pytorch.it_ptcnn_deconv import PTCNNDeconvolution


def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def demo(image):
    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)

    it_deconv = PTCNNDeconvolution(
        max_epochs=1500,
        patience=512,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
    )

    start = time.time()
    it_deconv.train(blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    aydin_deconvolved_image = it_deconv.translate(blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    lr = ImageTranslatorLRDeconv(
        psf_kernel=psf_kernel, max_num_iterations=20, backend="gputools"
    )
    lr.train(blurred_image)
    lr_deconvolved_image = lr.translate(blurred_image)

    image = numpy.clip(image, 0, 1)
    lr_deconvolved_image = numpy.clip(lr_deconvolved_image, 0, 1)
    aydin_deconvolved_image = numpy.clip(aydin_deconvolved_image, 0, 1)

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
    printscore(
        "n&b image:   ",
        psnr(image, blurred_image),
        spectral_mutual_information(image, blurred_image),
        mutual_information(image, blurred_image),
        ssim(image, blurred_image),
    )

    printscore(
        "lr deconv:    ",
        psnr(image, lr_deconvolved_image),
        spectral_mutual_information(image, lr_deconvolved_image),
        mutual_information(image, lr_deconvolved_image),
        ssim(image, lr_deconvolved_image),
    )

    printscore(
        "aydin deconv: ",
        psnr(image, aydin_deconvolved_image),
        spectral_mutual_information(image, aydin_deconvolved_image),
        mutual_information(image, aydin_deconvolved_image),
        ssim(image, aydin_deconvolved_image),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
        viewer.add_image(aydin_deconvolved_image, name='aydin_deconvolved_image')


image = dots()

demo(image)
