# flake8: noqa
import numpy as np
from skimage.data import camera

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import (
    normalise,
    add_noise,
    dots,
    lizard,
    pollen,
    characters,
    cropped_newyork,
)
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log
from aydin.util.nsi.nearest_self_image import nearest_self_image


def demo_nearest_self_image(image, display: bool = True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
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

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(self_image, name='self_image')
            viewer.add_image(self_noisy, name='self_noisy')
            viewer.add_image(denoised, name='denoised')


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
