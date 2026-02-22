"""Demo of Noise2Self perceptron denoising on a small 3D islet volume.

Loads a small cropped region of a 3D islet image, trains an
``ImageTranslatorFGR`` with ``PerceptronRegressor``, and visualizes
the result in napari.
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
    """Run Noise2Self perceptron denoising on a small 3D islet sub-volume."""
    # (3, 320, 865, 1014)
    image_path = examples_single.royerlab_hcr.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)
    image = image.squeeze()
    # image = image[1, 100:200, 400:500, 500:600]
    image = image[1, 100:300, 400:600, 400:600]
    print(image.shape)
    image = image.astype(numpy.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    generator = StandardFeatureGenerator(max_level=6, dtype=numpy.float16)
    regressor = PerceptronRegressor(depth=6, max_epochs=40, patience=10)
    it = ImageTranslatorFGR(generator, regressor)

    start = time.time()
    it.train(image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(image)
    stop = time.time()
    print(f"inference train: elapsed time:  {stop - start} ")

    print(image.shape)
    print(denoised.shape)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(denoised, name='denoised')
    napari.run()


if __name__ == "__main__":
    demo()
