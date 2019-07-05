import time

import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity

from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures


def demo(image):
    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(image, name='image')

        scales = [1, 3, 7, 15, 31]
        widths = [3, 3, 3, 3, 3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(num_leaves=128,
                                 n_estimators=1024,
                                 learning_rate=0.01,
                                 eval_metric='l1',
                                 early_stopping_rounds=None)

        it = ImageTranslatorClassic(generator, regressor)

        start = time.time()
        it.train(image, image)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')


image_path = examples_single.hyman_hela.get_path()
array, metadata = io.imread(image_path)
image = array[0:10, 25:35, 140:160, 130:170].astype(np.float32)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))
demo(image)
