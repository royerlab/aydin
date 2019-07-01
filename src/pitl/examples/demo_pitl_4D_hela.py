import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity

from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.regression.gbm import GBMRegressor
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic


def demo_pitl_4D_hela(image):


    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(image, name='image')

        scales = [1, 3, 7, 15, 31, 33]
        widths = [3, 3, 3,  3,  3,  3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                    kernel_scales=scales,
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(num_leaves=128,
                                 max_depth=8,
                                 n_estimators=1024,
                                 learning_rate=0.01,
                                 eval_metric='l1',
                                 early_stopping_rounds=None)

        it = ImageTranslatorClassic(generator, regressor)

        denoised = it.train(image, image)
        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')

        image_denoised = it.translate(image)
        viewer.add_image(rescale_intensity(image_denoised, in_range='image', out_range=(0, 1)), name='denoised_inference')


image_path = examples_single.hyman_hela.get_path()
array, metadata = io.imread(image_path)
image = array[0:10,25:35,140:160,130:170].astype(np.float32)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))
demo_pitl_4D_hela(image)
