# flake8: noqa
from os.path import join

import numpy
import numpy as np

from aydin.io import imread
from aydin.io.datasets import normalise, examples_single, examples_zipped
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log, lprint


def demo_butterworth_tribolium(noisy, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    lprint(noisy.shape)
    noisy = normalise(noisy.astype(np.float32))

    # noisy[:,:,0:10] = 0

    # if display:
    #     import napari
    #
    #     with napari.gui_qt():
    #         viewer = napari.Viewer()
    #         viewer.add_image(noisy, name='noisy')

    function, parameters, _ = calibrate_denoise_butterworth(
        noisy, isotropic=False, axes=(0, 1, 2)
    )
    denoised = function(noisy, **parameters)

    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')


if __name__ == "__main__":
    image, _ = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_stack.tif')
    )
    demo_butterworth_tribolium(image)
