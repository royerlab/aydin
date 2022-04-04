# flake8: noqa
import numpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import (
    normalise,
    add_noise,
    newyork,
    camera,
    characters,
    pollen,
    lizard,
    dots,
    cropped_newyork,
    dmel,
)
from aydin.it.classic_denoisers.tv import calibrate_denoise_tv
from aydin.util.log.log import Log


def demo_tv(image, do_add_noise=True, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(6)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image) if do_add_noise else image

    function, parameters, memreq = calibrate_denoise_tv(noisy)
    denoised = function(noisy, **parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("   noisy   :", psnr_noisy, ssim_noisy)
    print("tv denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()

    return ssim_denoised


if __name__ == "__main__":
    demo_tv(dmel(), do_add_noise=False)
    demo_tv(cropped_newyork())
    demo_tv(newyork())
    demo_tv(characters())
    demo_tv(pollen())
    demo_tv(lizard())
    demo_tv(dots())
    demo_tv(camera())
