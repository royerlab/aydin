# flake8: noqa
from skimage.data import camera

from aydin.io.datasets import dots, lizard, pollen, newyork, characters, examples_single
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import Log, lsection


def demo_representative_crop(
    image, crop_size=64000, search_mode: str = 'random', display: bool = True
):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    with lsection(f"Computing crop for image of shape: {image.shape}"):

        crop = representative_crop(
            image, crop_size=crop_size, search_mode=search_mode, display_crop=False
        )

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(crop, name='crop')

    assert crop.size <= int(crop_size * 1.25)


if __name__ == "__main__":

    demo_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 0, ...],
        crop_size=1_000_000,
    )

    demo_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 1, ...],
        crop_size=1_000_000,
    )

    demo_representative_crop(
        examples_single.royerlab_hcr.get_array().squeeze()[:, 2, ...],
        crop_size=1_000_000,
    )

    demo_representative_crop(
        examples_single.maitre_mouse.get_array(), crop_size=1_000_000, display=True
    )

    demo_representative_crop(
        examples_single.leonetti_arhgap21.get_array(), crop_size=1_000_000, display=True
    )

    demo_representative_crop(newyork())
    demo_representative_crop(camera())
    demo_representative_crop(characters())
    demo_representative_crop(pollen())
    demo_representative_crop(lizard())
    demo_representative_crop(dots())
