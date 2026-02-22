"""Demonstrate 2D Noise2Self denoising with CatBoost and variance stabilisation.

This demo simulates realistic camera noise (Poisson + readout) on test images
and compares several variance stabilisation transform (VST) modes (identity,
Box-Cox, Yeo-Johnson, quantile, Anscombe, log, sqrt) to evaluate their
effect on FGR denoising quality.
"""

# flake8: noqa
import time

import numpy as np

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import characters, newyork, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.highpass import HighpassTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image, name):
    """Denoise a 2D image using FGR with CatBoost under different VST modes.

    Parameters
    ----------
    image : numpy.ndarray
        Input 2D image.
    name : str
        Name used for labeling output visualizations.
    """
    Log.enable_output = True
    Log.set_log_max_depth(7)

    image = image.astype(np.float32) * 0.1
    noisy = simulate_camera_image(image)

    import napari

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
    napari.run()


def _denoise(noisy, mode):
    """Denoise using FGR with CatBoost and a specific VST mode.

    Parameters
    ----------
    noisy : numpy.ndarray
        Input noisy image.
    mode : str
        Variance stabilisation transform mode to apply before denoising.

    Returns
    -------
    numpy.ndarray
        Denoised image.
    """
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


if __name__ == "__main__":
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
