"""Demonstrate 3D spectral denoising on HeLa cell microscopy data.

This demo applies auto-calibrated spectral denoising to a single slice of
the Hyman HeLa dataset, visualizing noisy and denoised results with napari.
"""

# flake8: noqa

from aydin.io.datasets import examples_single
from aydin.it.classic_denoisers.spectral import calibrate_denoise_spectral
from aydin.util.log.log import Log


def demo_spectral_hela(display=True):
    """Denoise a HeLa slice using calibrated spectral denoising.

    Parameters
    ----------
    display : bool, optional
        Whether to display results with napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    noisy = examples_single.hyman_hela.get_array()[10]

    function, parameters, _ = calibrate_denoise_spectral(noisy)
    denoised = function(noisy, **parameters)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()


if __name__ == "__main__":
    demo_spectral_hela()
