"""Demo of J-invariance-based denoiser calibration.

Demonstrates the ``calibrate_denoiser`` function using J-invariance
to automatically find optimal Gaussian denoiser parameters, comparing
fast and smart optimiser modes.
"""

# flake8: noqa

from functools import partial

import numpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.io.datasets import add_noise, cropped_newyork, newyork, normalise
from aydin.it.classic_denoisers.gaussian import denoise_gaussian
from aydin.it.classic_denoisers.spectral import denoise_spectral
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser
from aydin.util.log.log import Log


def demo_j_invariant(image, optimiser_mode, display=True):
    """Calibrate a Gaussian denoiser using J-invariance and evaluate quality.

    Parameters
    ----------
    image : numpy.ndarray
        Input clean image (noise will be added synthetically).
    optimiser_mode : str
        Optimiser mode for calibration (e.g. ``'fast'`` or ``'smart'``).
    display : bool, optional
        Whether to display results in napari, by default True.

    Returns
    -------
    float
        SSIM between denoised and original image.
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
    best_parameters = calibrate_denoiser(
        crop,
        denoise_gaussian,
        mode=optimiser_mode,
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

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()
    return ssim_denoised


if __name__ == "__main__":
    demo_j_invariant(cropped_newyork(crop_amount=470), optimiser_mode='fast')
    demo_j_invariant(cropped_newyork(crop_amount=470), optimiser_mode='smart')
