# flake8: noqa

import numpy
import numpy as np
from numpy.random.mtrand import uniform
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import newyork, pollen, normalise, add_noise, lizard, dots
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image, name):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    # Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image, intensity=5, variance=0.01, sap=0, clip=False)
    noisier = noisy + (0.5) * (uniform(low=-1, high=1, size=image.shape))

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_line_features=True,
        include_spatial_features=True,
    )
    regressor = CBRegressor(
        patience=16, loss='l1', learning_rate=0.005, max_num_estimators=4096
    )

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.train(noisy, noisy, jinv=True)
    n2s_denoised = it.translate(noisy)

    it.train(noisier, noisy, jinv=False)
    denoised = it.translate(noisy)
    denoised_corrected = 2 * denoised - noisy

    # denoised2 = it.translate(it.translate(it.translate(denoised)))
    denoised2 = it.translate(denoised)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    n2s_denoised = numpy.clip(n2s_denoised, 0, 1)
    denoised_corrected = numpy.clip(denoised_corrected, 0, 1)
    denoised2 = numpy.clip(denoised2, 0, 1)

    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)

    psnr_n2s_denoised = psnr(image, n2s_denoised)
    ssim_n2s_denoised = ssim(image, n2s_denoised)

    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)

    psnr_denoised_corrected = psnr(image, denoised_corrected)
    ssim_denoised_corrected = ssim(image, denoised_corrected)

    psnr_denoised2 = psnr(image, denoised2)
    ssim_denoised2 = ssim(image, denoised2)

    print("noisy                             :", psnr_noisy, ssim_noisy)
    print(
        "denoised (classic_denoisers)                    :",
        psnr_n2s_denoised,
        ssim_n2s_denoised,
    )
    print("denoised (noiser3noise)           :", psnr_denoised, ssim_denoised)
    print(
        "denoised (noiser3noise corrected) :",
        psnr_denoised_corrected,
        ssim_denoised_corrected,
    )
    print("denoised (x2)                     :", psnr_denoised2, ssim_denoised2)

    Log.enable_output = False
    denoised_images = []
    for i in range(1, 32):
        psnr_denoised = psnr(image, numpy.clip(denoised, 0, 1))
        ssim_denoised = ssim(image, numpy.clip(denoised, 0, 1))
        print(
            f"denoised (x{i})                            :",
            psnr_denoised,
            ssim_denoised,
        )

        psnr_sslos = psnr(numpy.clip(n2s_denoised, 0, 1), numpy.clip(denoised, 0, 1))
        ssim_sslos = ssim(numpy.clip(n2s_denoised, 0, 1), numpy.clip(denoised, 0, 1))
        print(f"denoised ss loss(x{i})                     :", psnr_sslos, ssim_sslos)

        denoised_images.append(denoised)
        denoised = it.translate(denoised)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(noisier, name='noisier')
        viewer.add_image(n2s_denoised, name='denoised (classic_denoisers)')
        viewer.add_image(denoised, name='denoised (noiser3noise)')
        viewer.add_image(denoised_corrected, name='denoised (noiser3noise corrected)')
        viewer.add_image(numpy.stack(denoised_images), name=f'denoised images')


camera_image = camera()
demo(camera_image, "camera")
lizard_image = lizard()
demo(lizard_image, "lizard")
pollen_image = pollen()
demo(pollen_image, "pollen")
dots_image = dots()
demo(dots_image, "dots")
newyork_image = newyork()
demo(newyork_image, "newyork")
