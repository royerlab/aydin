# flake8: noqa
import time

import numpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, add_noise, dots, camera, cropped_newyork
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image, name, do_add_noise=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    Log.enable_output = True

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image) if do_add_noise else image

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
        include_median_features=True,
        include_dct_features=True,
        num_sinusoidal_features=4,
        include_random_conv_features=True,
    )

    regressor = CBRegressor(patience=256, gpu=True, min_num_estimators=1024)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("noisy   :", psnr_noisy, ssim_noisy)
    print("denoised:", psnr_denoised, ssim_denoised)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')


camera_image = camera()
demo(camera_image, "camera")
# lizard_image = lizard()
# demo(lizard_image, "lizard")
# pollen_image = pollen()
# demo(pollen_image, "pollen")
# newyork_image = newyork()
# demo(newyork_image, "newyork")
newyork_image = cropped_newyork()
demo(newyork_image, "newyork")
dots_image = dots()
demo(dots_image, "dots")
