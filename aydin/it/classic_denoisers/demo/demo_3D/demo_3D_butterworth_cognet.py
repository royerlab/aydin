"""Demonstrate 3D Butterworth denoising on CogNet nanotube time-lapse data.

This demo applies auto-calibrated Butterworth low-pass filtering in z-yx mode
to a CogNet nanotube 400fps dataset, visualizing the noisy and denoised
volumes with napari.
"""

# flake8: noqa
import numpy
import numpy as np

from aydin.io.datasets import examples_single, normalise
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log, aprint


def demo_butterworth_cognet(noisy, display=True):
    """Denoise a 3D CogNet volume using calibrated Butterworth filtering.

    Parameters
    ----------
    noisy : numpy.ndarray
        Input 3D noisy image array.
    display : bool, optional
        Whether to display results with napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    aprint(noisy.shape)
    noisy = normalise(noisy.astype(np.float32))

    # noisy[:,:,0:10] = 0

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(noisy, name='noisy')
        napari.run()
    function, parameters, _ = calibrate_denoise_butterworth(
        noisy, mode='z-yx', axes=(0, 1, 2)
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
    array = examples_single.cognet_nanotube_400fps.get_array()
    demo_butterworth_cognet(array)
