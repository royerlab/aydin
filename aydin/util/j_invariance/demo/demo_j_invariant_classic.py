# flake8: noqa

import numpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import normalise, add_noise, newyork
from aydin.it.classic_denoisers.gaussian import denoise_gaussian
from aydin.it.classic_denoisers.spectral import denoise_spectral
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import Log


def demo_j_invariant_classic(image, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    # obtain representative crop, to speed things up...
    crop = representative_crop(noisy)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': (0.0, 1.0), 'truncate': [4, 1]}

    # Calibrate denoiser
    best_parameters = calibrate_denoiser_classic(
        crop,
        denoise_gaussian,
        denoise_parameters=parameter_ranges,
        display_images=False,
    )

    denoised = denoise_gaussian(noisy, **best_parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("         noisy   :", psnr_noisy, ssim_noisy)
    print("spectral denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')

    assert ssim_denoised > 0.64

    return ssim_denoised


if __name__ == "__main__":
    demo_j_invariant_classic(newyork())
