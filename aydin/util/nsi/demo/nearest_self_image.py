"""Demo of nearest-self-image denoising approach.

Demonstrates the ``nearest_self_image`` utility by computing a
self-similar version of a noisy image and using it as a training
target for FGR-based denoising on various test images.
"""

# flake8: noqa
import numpy as np
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import (
    add_noise,
    characters,
    cropped_newyork,
    dots,
    lizard,
    normalise,
    pollen,
)
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log
from aydin.util.nsi.nearest_self_image import nearest_self_image


def demo_nearest_self_image(image, display: bool = True):
    """Denoise an image using the nearest-self-image approach.

    Parameters
    ----------
    image : numpy.ndarray
        Input clean image (noise will be added synthetically).
    display : bool, optional
        Whether to display results in napari, by default True.
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    self_image = nearest_self_image(image)
    self_noisy = nearest_self_image(noisy)

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )

    regressor = CBRegressor(patience=32, gpu=True, min_num_estimators=1024)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.transforms_list.append(RangeTransform())
    it.transforms_list.append(PaddingTransform())

    it.train(self_noisy, noisy, jinv=False)

    denoised = it.translate(noisy)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(self_image, name='self_image')
        viewer.add_image(self_noisy, name='self_noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()


if __name__ == "__main__":
    newyork_image = cropped_newyork()  # [0:64, 0:64]
    demo_nearest_self_image(newyork_image)
    camera_image = camera()
    demo_nearest_self_image(camera_image)
    characters_image = characters()
    demo_nearest_self_image(characters_image)
    pollen_image = pollen()
    demo_nearest_self_image(pollen_image)
    lizard_image = lizard()
    demo_nearest_self_image(lizard_image)
    dots_image = dots()
    demo_nearest_self_image(dots_image)
