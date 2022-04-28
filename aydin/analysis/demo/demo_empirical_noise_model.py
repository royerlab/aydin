# flake8: noqa
from numpy.random.mtrand import randint

from aydin.analysis.empirical_noise_model import (
    distill_noise_model,
    sample_noise_from_model,
)
from aydin.io.datasets import camera


def demo_noise_model():
    clean_image = camera()

    noise = randint(0, 60, size=clean_image.size).reshape(*clean_image.shape)
    noisy_image = clean_image + noise

    noise_model = distill_noise_model(clean_image, noisy_image)

    noisy_image_sampled = sample_noise_from_model(clean_image, noise_model)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(clean_image, name='clean_image')
        viewer.add_image(noisy_image, name='noisy_image')
        viewer.add_image(noisy_image_sampled, name='noisy_image_sampled')
        viewer.add_image(noise_model, name='noise_model')


if __name__ == "__main__":
    demo_noise_model()
