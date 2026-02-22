"""Demonstrate 3D batched Noise2Self denoising with CatBoost on HeLa data.

This demo applies self-supervised FGR denoising with CatBoost to the Hyman
HeLa 4D dataset, treating the first axis as a batch dimension so that each
time point is denoised independently, with napari visualization.
"""

# flake8: noqa
import time

import numpy
from skimage.exposure import rescale_intensity

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def demo():
    """Denoise 3D HeLa volumes in batch mode using FGR with CatBoost."""
    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    # image = image[0:10, 15:35, 130:167, 130:177]
    image = image.astype(numpy.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    batch_dims = (True, False, False, False)

    generator = StandardFeatureGenerator()

    regressor = CBRegressor()

    it = ImageTranslatorFGR(generator, regressor, balance_training_data=True)

    start = time.time()
    it.train(image, image, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(image, batch_axes=batch_dims)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(
        rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
        name='denoised',
    )
    napari.run()


if __name__ == "__main__":
    demo()
