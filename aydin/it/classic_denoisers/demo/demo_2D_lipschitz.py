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
)
from aydin.it.classic_denoisers.gaussian import calibrate_denoise_gaussian
from aydin.it.classic_denoisers.lipschitz import calibrate_denoise_lipschitz
from aydin.util.log.log import Log


def demo_lipschitz(image, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image, intensity=64, variance=0.001, sap=0.2)
    noisy = add_noise(noisy, intensity=128, variance=0.01, sap=0)

    function, parameters, memreq = calibrate_denoise_lipschitz(noisy)
    denoised = function(noisy, **parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("         noisy   :", psnr_noisy, ssim_noisy)
    print("gaussian denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')

    return ssim_denoised


if __name__ == "__main__":
    demo_lipschitz(newyork())
    demo_lipschitz(characters())
    demo_lipschitz(pollen())
    demo_lipschitz(lizard())
    demo_lipschitz(dots())
    demo_lipschitz(camera())
