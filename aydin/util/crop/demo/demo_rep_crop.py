# flake8: noqa
import numpy
import numpy as np
from skimage.data import camera

from aydin.io.datasets import (
    normalise,
    add_noise,
    dots,
    lizard,
    pollen,
    newyork,
    characters,
    examples_single,
)
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import Log, lsection


def demo_representative_crop(image, fast_mode: bool = False, display: bool = True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    with lsection(f"Computing crop for image of shape: {image.shape}"):
        crop_size = 64000
        crop = representative_crop(
            image, crop_size=crop_size, fast_mode=fast_mode, display_crop=False
        )

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(crop, name='crop')

    assert crop.size <= int(crop_size * 1.05)


if __name__ == "__main__":

    islet = examples_single.royerlab_hcr.get_array().squeeze()[1]
    demo_representative_crop(islet, fast_mode=True)

    demo_representative_crop(newyork())
    demo_representative_crop(camera())
    demo_representative_crop(characters())
    demo_representative_crop(pollen())
    demo_representative_crop(lizard())
    demo_representative_crop(dots())
