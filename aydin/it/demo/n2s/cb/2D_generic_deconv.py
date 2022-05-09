# flake8: noqa

import numpy
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.find_kernel import compute_relative_blur_kernel
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import newyork, pollen, normalise, add_noise, lizard, dots
from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def demo(image, name):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    # Log.set_log_max_depth(5)

    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=False,
        include_fine_features=True,
        include_line_features=True,
        include_spatial_features=True,
    )
    regressor = CBRegressor(patience=16)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.train(image, image)
    denoised_clean = it.translate(image)

    computed_kernel = compute_relative_blur_kernel(denoised_clean, image)

    psf_kernel = numpy.asarray([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
    psf_kernel = psf_kernel / psf_kernel.sum()

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(computed_kernel), name='computed_kernel')
        viewer.add_image(normalise(psf_kernel), name='psf_kernel')

    it.train(noisy, noisy)
    denoised = it.translate(noisy)

    lr = ImageTranslatorLRDeconv(
        psf_kernel=computed_kernel, max_num_iterations=3, backend='cupy'
    )
    lr.train(denoised, denoised)
    denoised_deconv = lr.translate(denoised)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    denoised_deconv = numpy.clip(denoised_deconv, 0, 1)

    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)

    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)

    psnr_denoised_deconv = psnr(image, denoised_deconv)
    ssim_denoised_deconv = ssim(image, denoised_deconv)

    print("noisy          :", psnr_noisy, ssim_noisy)
    print("denoised       :", psnr_denoised, ssim_denoised)
    print("denoised_deconv:", psnr_denoised_deconv, ssim_denoised_deconv)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            normalise(numpy.log1p(numpy.absolute(computed_kernel))),
            name='computed_kernel',
        )
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised), name='denoised')
        viewer.add_image(normalise(denoised_deconv), name='denoised_deconv')


if __name__ == "__main__":
    newyork_image = newyork()
    demo(newyork_image, "newyork")
    camera_image = camera()
    demo(camera_image, "camera")
    lizard_image = lizard()
    demo(lizard_image, "lizard")
    pollen_image = pollen()
    demo(pollen_image, "pollen")
    dots_image = dots()
    demo(dots_image, "dots")
