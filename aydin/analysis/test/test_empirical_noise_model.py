import numpy
from numpy.random.mtrand import randint

from aydin.analysis.empirical_noise_model import (
    distill_noise_model,
    sample_noise_from_model,
)
from aydin.io.datasets import camera


def test_noise_model():
    clean_image = camera()

    noise = randint(0, 60, size=clean_image.size).reshape(*clean_image.shape)
    noisy_image = clean_image + noise

    noise_model = distill_noise_model(clean_image, noisy_image)

    noisy_image_sampled = sample_noise_from_model(clean_image, noise_model)

    diff = numpy.abs(noisy_image - noisy_image_sampled)

    assert (diff < 60).all()
