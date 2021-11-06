import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from scipy.stats import entropy

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.nn.util.data_util import random_sample_patches


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


# Test with garder 3D image
def test_random_sample_patch_3D():
    image_path = examples_single.hyman_hela.get_path()
    image0, metadata = io.imread(image_path)
    print(image0.shape)
    image0 = n(image0.squeeze()[0:32, :, 200:300, 200:300])

    image0 = numpy.expand_dims(image0[1:2], -1)
    tile_size = (32, 32, 32)
    num_tile = 100
    adoption_rate = 0.2
    input_data = random_sample_patches(image0, tile_size, num_tile, adoption_rate)

    # Extract patched images
    img_patch = []
    for i in input_data:
        img_patch.append(image0[i])
    img_patch = numpy.vstack(img_patch)
    # Entropy of the whole image
    hist, _ = numpy.histogram(image0, range=(0, 1), bins=255, density=True)
    entropy_whole = entropy(hist)

    # Entropy of sampled areas.
    hist, _ = numpy.histogram(img_patch, range=(0, 1), bins=255, density=True)
    entropy_smpl = entropy(hist)

    ent_ratio = entropy_smpl / entropy_whole

    assert ent_ratio >= 1.0


# Test with Cameraman
def test_random_sample_patch_2D():
    image0 = camera().astype(numpy.float32)
    image0 = numpy.expand_dims(numpy.expand_dims(n(image0), -1), 0)
    patch_size = (64, 64)
    num_patch = 500
    adoption_rate = 0.5
    input_data = random_sample_patches(image0, patch_size, num_patch, adoption_rate)

    # Extract patched images
    img_patch = []
    for i in input_data:
        img_patch.append(image0[i])
    img_patch = numpy.vstack(img_patch)
    # Entropy of the whole image
    hist, _ = numpy.histogram(image0, range=(0, 1), bins=255, density=True)
    entropy_whole = entropy(hist)

    # Entropy of sampled areas.
    hist, _ = numpy.histogram(img_patch, range=(0, 1), bins=255, density=True)
    entropy_smpl = entropy(hist)

    ent_ratio = entropy_smpl / entropy_whole

    assert ent_ratio >= 1.01


def test_random_sample_patch_2D_1patch():
    """
    Test whether random_sample_patches generates 1 image when patch size is same as input size.
    """
    image0 = camera().astype(numpy.float32)
    image0 = numpy.expand_dims(numpy.expand_dims(n(image0), -1), 0)
    patch_size = (512, 512)
    num_patch = 500
    adoption_rate = 0.5
    input_data = random_sample_patches(image0, patch_size, num_patch, adoption_rate)

    # Extract patched images
    img_patch = []
    for i in input_data:
        img_patch.append(image0[i])
    img_patch = numpy.vstack(img_patch)

    assert img_patch.shape == image0.shape
