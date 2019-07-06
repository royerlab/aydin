import time

from napari.util import app_context
from skimage.exposure import rescale_intensity

from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor
from pitl.features.mcfocl import MultiscaleConvolutionalFeatures


def demo(image):
    from napari import Viewer
    with app_context():
        viewer = Viewer()
        viewer.add_image(image, name='image')

        level = 1
        scales = [1, 3, 7, 15, 31]
        widths = [3, 3, 3, 3, 3]

        generator = MultiscaleConvolutionalFeatures(kernel_widths=widths[:level],
                                                    kernel_scales=scales[:level],
                                                    exclude_center=False
                                                    )

        regressor = GBMRegressor(num_leaves=128,
                                 n_estimators=128,
                                 learning_rate=0.01,
                                 metric='l1',
                                 early_stopping_rounds=None)

        it = ImageTranslatorClassic(generator, regressor)

        batch_dims = (True, False, False, False)

        start = time.time()
        it.train(image, image, batch_dims=batch_dims, batch_size=100 * 1e6)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer.add_image(rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised')


# (3, 320, 865, 1014)
image_path = examples_single.gardner_org.get_path()
array, metadata = io.imread(image_path)
demo(array)
