# flake8: noqa

from aydin.io.datasets import examples_single
from aydin.it.classic_denoisers.dictionary_learned import (
    calibrate_denoise_dictionary_learned,
)

from aydin.util.log.log import Log


def demo_dict_learn_hela(display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    noisy = examples_single.hyman_hela.get_array()[10]

    function, parameters, _ = calibrate_denoise_dictionary_learned(
        noisy, dictionary_type='learned', display_dictionary=display
    )
    denoised = function(noisy, **parameters)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')


if __name__ == "__main__":
    demo_dict_learn_hela()
