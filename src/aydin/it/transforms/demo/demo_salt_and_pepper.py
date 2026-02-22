"""Demo of salt-and-pepper noise correction transform.

Demonstrates the ``SaltPepperTransform`` on various test images with
synthetic salt-and-pepper noise, comparing the result to median
filtering.
"""

# flake8: noqa
from scipy.ndimage import median_filter

from aydin.io.datasets import (
    add_noise,
    camera,
    characters,
    dots,
    lizard,
    newyork,
    normalise,
    pollen,
)
from aydin.it.transforms.salt_pepper import SaltPepperTransform
from aydin.util.log.log import Log


def demo_salt_and_pepper(image=newyork()):
    """Apply salt-and-pepper correction and compare with median filtering.

    Parameters
    ----------
    image : numpy.ndarray, optional
        Input test image, by default the New York image.
    """
    Log.override_test_exclusion = True
    Log.enable_output = True

    image = normalise(image)
    noisy = add_noise(image, intensity=64, variance=0.001, sap=0.1)
    noisy = add_noise(noisy, intensity=128, variance=0.01, sap=0)
    noisy = add_noise(noisy, intensity=256, variance=0.001, sap=0.1)

    bpc = SaltPepperTransform()

    corrected = bpc.preprocess(noisy)

    median = median_filter(noisy, size=3)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(median, name='median_filtered')
    viewer.add_image(corrected, name='corrected')
    napari.run()


if __name__ == "__main__":
    demo_salt_and_pepper(newyork())
    demo_salt_and_pepper(characters())
    demo_salt_and_pepper(pollen())
    demo_salt_and_pepper(lizard())
    demo_salt_and_pepper(dots())
    demo_salt_and_pepper(camera())
