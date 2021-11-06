# flake8: noqa
import time

import numpy as np
from skimage.exposure import rescale_intensity

from aydin.io import io
from aydin.io.datasets import examples_single, add_noise
from aydin.it.cnn import ImageTranslatorCNN

image_path = examples_single.royerlab_hcr.get_path()
image, metadata = io.imread(image_path)  # (13, 520, 696)
print(image.shape)
# image = image[0:10, 15:35, 130:167, 130:177]
image = image[:, :, :, :, 100:300, 100:600, 200:700, :]
image = rescale_intensity(image.astype(np.float32), in_range='image', out_range=(0, 1))
noisy = add_noise(image)
max_epochs = 10

it = ImageTranslatorCNN(
    training_architecture='random',
    model_architecture='jinet',
    nb_unet_levels=3,
    batch_norm='instance',  # None,  #
    activation='ReLU',
    mask_size=3,
    max_epochs=max_epochs,
)

it.verbose = 1

start = time.time()
it.train(noisy, image, batch_axes=metadata.batch_axes)
stop = time.time()
print(f"Training: elapsed time:  {stop - start} ")

image = image[:, :, :, 1:2, :, :, :, :]

start = time.time()
denoised = it.translate(
    image,
    batch_axes=metadata.batch_axes,
    tile_size=128,  # image.shape[1:-1],  # [12, 12, 12],  # min(image.shape[1:-1])
)
stop = time.time()
print(f"inference: elapsed time:  {stop - start} ")
import napari

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(
        rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised'
    )

# demo(image)
