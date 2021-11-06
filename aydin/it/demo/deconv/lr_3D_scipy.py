# flake8: noqa
import time

import numpy
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, examples_single, add_blur_3d
from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.util.log.log import Log


def demo(image):
    Log.enable_output = True
    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_3d(image, xy_size=17, z_size=31)

    it = ImageTranslatorLRDeconvScipy(
        psf_kernel=psf_kernel, max_num_iterations=100, clip=False
    )

    start = time.time()
    it.train(blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    lr_deconvolved_image = it.translate(blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")

    def printscore(header, val1, val2, val3, val4):
        print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")

    printscore(
        "n&b image",
        psnr(image, blurred_image),
        spectral_mutual_information(image, blurred_image),
        mutual_information(image, blurred_image),
        ssim(image, blurred_image),
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
        viewer.add_image(psf_kernel, name='psf_kernel')
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')


image = examples_single.janelia_flybrain.get_array()
# image = image[:, 1, :, :].astype(numpy.float32)
image = image[:, 1, 228:-228, 228:-228].astype(numpy.float32)
# image = image[100:165, 133:217, 150:231]
image = rescale_intensity(image, in_range='image', out_range=(0, 1))

demo(image)
