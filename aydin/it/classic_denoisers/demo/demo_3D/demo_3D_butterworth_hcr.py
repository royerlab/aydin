# flake8: noqa
import numpy
import numpy as np

from aydin.io.datasets import normalise, examples_single
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log, lprint


def demo_lowpass_islet(noisy, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    lprint(noisy.shape)
    noisy = normalise(noisy.astype(np.float32))

    function, parameters, _ = calibrate_denoise_butterworth(
        noisy, mode='z-yx', axes=(0, 1, 2)
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
    hcr = examples_single.royerlab_hcr.get_array().squeeze()
    hcr = hcr[:, 1, :500, :500]
    demo_lowpass_islet(hcr)
