"""Demo of Noise2Self perceptron denoising on 4D HeLa cell data.

Loads a 4D HeLa dataset, trains an ``ImageTranslatorFGR`` with
``PerceptronRegressor`` with balanced training data, and visualizes
the denoised result in napari.
"""

# flake8: noqa
import time

import numpy
from skimage.exposure import rescale_intensity

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.perceptron import PerceptronRegressor


def demo():
    """Run Noise2Self perceptron denoising on the 4D HeLa cell dataset."""
    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)
    # image = image[0:10, 15:35, 130:167, 130:177]
    image = image.astype(numpy.float16)
    print(image.shape)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    generator = StandardFeatureGenerator()

    regressor = PerceptronRegressor()

    it = ImageTranslatorFGR(generator, regressor, balance_training_data=True)

    start = time.time()
    it.train(image, image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(image)
    stop = time.time()
    print(f"inference train: elapsed time:  {stop - start} ")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(denoised, name='denoised')
    napari.run()


if __name__ == "__main__":
    demo()
