# flake8: noqa
import numpy
from numpy.random import normal
from skimage.data import camera

from aydin.io.datasets import (
    dots,
    lizard,
    pollen,
    newyork,
    characters,
    examples_single,
    normalise,
)
from aydin.util.crop.super_fast_rep_crop import super_fast_representative_crop
from aydin.util.log.log import Log, lsection, lprint


def demo_super_fast_representative_crop(image, crop_size=64000, display: bool = True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(numpy.float32))
    image += 0.1 * normal(size=image.shape, scale=0.1)

    def _crop_fun():
        return super_fast_representative_crop(
            image, crop_size=crop_size, display_crop=False
        )

    # Warmup (numba compilation)
    # _crop_fun()

    with lsection(f"Computing crop for image of shape: {image.shape}"):
        # for _ in range(10):
        crop = _crop_fun()

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(crop, name='crop')
        napari.run()

    lprint(f"Crop size requested: {crop_size} obtained: {crop.size}")

    assert crop.size >= int(crop_size * 0.5) and crop.size <= int(crop_size * 2)


if __name__ == "__main__":

    demo_super_fast_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze(), crop_size=1_000_000
    )

    demo_super_fast_representative_crop(
        examples_single.maitre_mouse.get_array(), crop_size=1_000_000
    )

    demo_super_fast_representative_crop(
        examples_single.leonetti_arhgap21.get_array(), crop_size=1_000_000
    )

    demo_super_fast_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 0, ...],
        crop_size=1_000_000,
    )

    demo_super_fast_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 1, ...],
        crop_size=1_000_000,
    )

    demo_super_fast_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 2, ...],
        crop_size=1_000_000,
    )

    demo_super_fast_representative_crop(newyork())
    demo_super_fast_representative_crop(camera())
    demo_super_fast_representative_crop(characters())
    demo_super_fast_representative_crop(pollen())
    demo_super_fast_representative_crop(lizard())
    demo_super_fast_representative_crop(dots())
