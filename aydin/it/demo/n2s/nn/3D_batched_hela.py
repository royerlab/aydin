# flake8: noqa
import time

import numpy as np
from skimage.exposure import rescale_intensity

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.nn import NNRegressor


def demo():
    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    # image = image[0:10, 15:35, 130:167, 130:177]
    image = image.astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    batch_dims = (True, False, False, False)

    generator = StandardFeatureGenerator(max_level=5)

    regressor = NNRegressor()

    it = ImageTranslatorFGR(generator, regressor)

    start = time.time()
    it.train(image, image, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(image, batch_axes=batch_dims)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )


demo()
