# flake8: noqa
from scipy.ndimage import median_filter

from aydin.io.datasets import (
    normalise,
    newyork,
    characters,
    pollen,
    lizard,
    dots,
    camera,
    add_noise,
)
from aydin.it.transforms.salt_pepper import SaltPepperTransform
from aydin.util.log.log import Log


def demo_salt_and_pepper(image=newyork()):
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

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(median, name='median_filtered')
        viewer.add_image(corrected, name='corrected')


demo_salt_and_pepper(newyork())
demo_salt_and_pepper(characters())
demo_salt_and_pepper(pollen())
demo_salt_and_pepper(lizard())
demo_salt_and_pepper(dots())
demo_salt_and_pepper(camera())
