"""Demo of Noise2Self CNN denoising on batched 3D HeLa cell data.

Loads a 4D HeLa dataset, treats the first axis as a batch dimension,
and trains an ``ImageTranslatorCNNTorch`` for self-supervised 3D denoising.
"""

import time

import numpy as np
from skimage.exposure import rescale_intensity

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.cnn_torch import ImageTranslatorCNNTorch


def demo(image, max_epochs=10):
    """Run Noise2Self CNN denoising on batched 3D HeLa image data.

    Parameters
    ----------
    image : numpy.ndarray
        4D array where the first axis is the batch dimension.
    max_epochs : int, optional
        Maximum number of training epochs.
    """
    batch_dims = (True, False, False, False)

    it = ImageTranslatorCNNTorch(
        training_architecture='random',
        nb_unet_levels=2,
        batch_norm='instance',  # None,  #
        activation='ReLU',
        mask_size=3,
        # tile_size=128,
        # total_num_patches=10,
        max_epochs=max_epochs,
    )
    it.verbose = 1

    start = time.time()
    it.train(image, image, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(
        image, batch_axes=batch_dims, tile_size=100  # min(image.shape[1:-1])
    )
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
    image_path = examples_single.hyman_hela.get_path()
    array, metadata = io.imread(image_path)
    # array = array[0:10, 15:35, 130:167, 130:177]
    array = array[:, :, 100:300, 100:300].astype(np.float32)
    array = rescale_intensity(array, in_range='image', out_range=(0, 1))
    demo(array)
