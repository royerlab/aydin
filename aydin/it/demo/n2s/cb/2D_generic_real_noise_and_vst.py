# flake8: noqa
import time

import numpy as np

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import newyork, characters, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.highpass import HighpassTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image, name):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(7)

    image = image.astype(np.float32) * 0.1
    noisy = simulate_camera_image(image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')

        for mode in [
            'identity',
            'box-cox',
            'yeo-johnson',
            'quantile',
            'anscomb',
            'log',
            'sqrt',
        ]:
            denoised = _denoise(noisy, mode)
            viewer.add_image(denoised, name=f'denoised_{mode}')


def _denoise(noisy, mode):
    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=False,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = CBRegressor(patience=32, gpu=True, min_num_estimators=1024)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.transforms_list.append(VarianceStabilisationTransform(mode=mode))
    it.transforms_list.append(RangeTransform())
    it.transforms_list.append(HighpassTransform())
    # it.transforms_list.append(PaddingTransform())
    print("training starts")
    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")
    # in case of batching we have to do this:
    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")
    return denoised


# lizard_image = lizard()
# demo(lizard_image, "lizard")
pollen_image = pollen()
demo(pollen_image, "pollen")
newyork_image = newyork()
demo(newyork_image, "newyork")
characters_image = characters()
demo(characters_image, "characters")


# dots_image = dots()
# demo(dots_image, "dots")
