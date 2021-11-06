# flake8: noqa
import numpy

from aydin.io.datasets import newyork
from aydin.util.patch_transform.patch_transform import (
    extract_patches_nd,
    reconstruct_from_nd_patches,
)


def test_patch_transform():
    _round_trip(7, display=False)
    _round_trip(8, display=False)


def _round_trip(patch_size: int, display: bool):
    # Image:
    image = newyork()
    # First we apply the patch transform:
    patches = extract_patches_nd(image, patch_size=patch_size)
    # Transform back from patches to image:
    reconstructed_image = reconstruct_from_nd_patches(patches, image.shape)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(patches, name='patches')
            viewer.add_image(reconstructed_image, name='reconstructed_image')
        assert numpy.allclose(image, reconstructed_image, atol=1)
