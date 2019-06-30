import numpy as np
from napari.util import app_context
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from pitl.io import io
from pitl.io.datasets import examples_single
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.gbm import GBMRegressor


def demo_pitl_3D_batched_hela(image):


    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(image, name='image')

        scales = [1, 3, 7, 15, 31, 63]
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

        it = ImageTranslator(generator, regressor)

        denoised = it.train(image, image, batch_dims=(True, True, False, False))
        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')

        image_denoised = it.translate(image)
        viewer.add_image(rescale_intensity(image_denoised, in_range='image', out_range=(0, 1)), name='denoised_inference')


image_path = examples_single.get_path(*examples_single.hyman_hela)
array, metadata = io.imread(image_path)
image = array.astype(np.float32)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))
demo_pitl_3D_batched_hela(image)
