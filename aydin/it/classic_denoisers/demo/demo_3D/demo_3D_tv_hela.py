# flake8: noqa

from aydin.io.datasets import normalise, examples_single
from aydin.it.classic_denoisers.tv import calibrate_denoise_tv
from aydin.util.log.log import Log


def demo_tv_hela(display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    noisy = examples_single.hyman_hela.get_array()[10]
    noisy = normalise(noisy)

    function, parameters, _ = calibrate_denoise_tv(noisy, display_images=True)
    denoised = function(noisy, **parameters)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')


if __name__ == "__main__":
    demo_tv_hela()
