# flake8: noqa
import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import (
    normalise,
    add_noise,
    dots,
    newyork,
    lizard,
    pollen,
    characters,
    cropped_newyork,
)
from aydin.it.classic_denoisers.bmnd import calibrate_denoise_bmnd
from aydin.it.classic_denoisers.spectral import calibrate_denoise_spectral
from aydin.util.log.log import Log


def demo_bmnd(image, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    function, parameters, memreq = calibrate_denoise_bmnd(noisy)
    denoised = function(noisy, patch_size=5, block_depth=64, **parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("         noisy   :", psnr_noisy, ssim_noisy)
    print("bmnd     denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')

    return ssim_denoised


if __name__ == "__main__":
    newyork_image = cropped_newyork()
    demo_bmnd(newyork_image)
    characters_image = characters()
    demo_bmnd(characters_image)
    pollen_image = pollen()
    demo_bmnd(pollen_image)
    lizard_image = lizard()
    demo_bmnd(lizard_image)
    dots_image = dots()
    demo_bmnd(dots_image)
    camera_image = camera()
    demo_bmnd(camera_image)
