"""Demonstrate 3D Butterworth denoising on HeLa cell microscopy data.

This demo applies auto-calibrated Butterworth low-pass filtering to the
Hyman HeLa 4D dataset (treating all four axes), visualizing the denoised
result with napari.
"""

# flake8: noqa
import numpy
import numpy as np

from aydin.io.datasets import examples_single, normalise
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log, aprint


def demo_butterworth_cognet(noisy, display=True):
    """Denoise a 3D/4D HeLa volume using calibrated Butterworth filtering.

    Parameters
    ----------
    noisy : numpy.ndarray
        Input noisy image array from HeLa dataset.
    display : bool, optional
        Whether to display results with napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    aprint(noisy.shape)
    noisy = normalise(noisy.astype(np.float32))

    function, parameters, _ = calibrate_denoise_butterworth(
        noisy, mode='z-yx', axes=(0, 1, 2, 3)
    )
    denoised = function(noisy, **parameters)

    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()


if __name__ == "__main__":
    hyman_hela = examples_single.hyman_hela.get_array()
    demo_butterworth_cognet(hyman_hela)
