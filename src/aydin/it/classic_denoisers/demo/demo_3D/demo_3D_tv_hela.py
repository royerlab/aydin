"""Demonstrate 3D total variation denoising on HeLa cell microscopy data.

This demo applies auto-calibrated total variation (TV) denoising to a single
slice of the Hyman HeLa dataset, visualizing noisy and denoised results
with napari.
"""

# flake8: noqa

from aydin.io.datasets import examples_single, normalise
from aydin.it.classic_denoisers.tv import calibrate_denoise_tv
from aydin.util.log.log import Log


def demo_tv_hela(display=True):
    """Denoise a HeLa slice using calibrated total variation denoising.

    Parameters
    ----------
    display : bool, optional
        Whether to display results with napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    noisy = examples_single.hyman_hela.get_array()[10]
    noisy = normalise(noisy)

    function, parameters, _ = calibrate_denoise_tv(noisy, display_images=True)
    denoised = function(noisy, **parameters)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()


if __name__ == "__main__":
    demo_tv_hela()
