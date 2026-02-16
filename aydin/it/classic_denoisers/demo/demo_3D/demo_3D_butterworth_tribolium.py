"""Demonstrate 3D Butterworth denoising on Tribolium embryo data.

This demo applies auto-calibrated Butterworth low-pass filtering in z-yx mode
to a CARE Tribolium training stack, visualizing the noisy and denoised
volumes with napari.
"""

# flake8: noqa
from os.path import join

import numpy
import numpy as np

from aydin.io import imread
from aydin.io.datasets import examples_zipped, normalise
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log, aprint


def demo_butterworth_tribolium(noisy, display=True):
    """Denoise a 3D Tribolium volume using calibrated Butterworth filtering.

    Parameters
    ----------
    noisy : numpy.ndarray
        Input 3D noisy image array from Tribolium dataset.
    display : bool, optional
        Whether to display results with napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    aprint(noisy.shape)
    noisy = normalise(noisy.astype(np.float32))

    # noisy[:,:,0:10] = 0

    # if display:
    #     import napari
    #
    #     with napari.gui_qt():
    #         viewer = napari.Viewer()
    #         viewer.add_image(noisy, name='noisy')

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
    image, _ = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_stack.tif')
    )
    demo_butterworth_tribolium(image)
