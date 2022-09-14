# flake8: noqa
# import pytest
import pytest
import matplotlib.pyplot as plt
from numpy.random.mtrand import normal

from aydin.analysis.fsc import fsc
from aydin.io.datasets import camera, normalise
from aydin.util.log.log import lprint, Log


def demo_fsc(display: bool = True):
    Log.enable_output = True

    clean_image = normalise(camera())

    noise1 = normal(size=clean_image.size).reshape(*clean_image.shape)
    noise2 = normal(size=clean_image.size).reshape(*clean_image.shape)

    noisy_image_1 = clean_image + noise1
    noisy_image_2 = clean_image + noise2

    correlations = fsc(noisy_image_1, noisy_image_2)

    plt.plot(correlations)
    plt.title('noisy1, noisy2')
    plt.show()

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(clean_image, name='clean_image')
        viewer.add_image(noisy_image_1, name='noisy_image_1')
        viewer.add_image(noisy_image_2, name='noisy_image_2')
        napari.run()


if __name__ == "__main__":
    demo_fsc()
