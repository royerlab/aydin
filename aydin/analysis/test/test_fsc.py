import pytest
from numpy.random.mtrand import normal

from aydin.analysis.fsc import shell_sum, fsc
from aydin.io.datasets import camera, normalise, newyork


@pytest.mark.parametrize("image, length", [(camera(), 363), (newyork(), 725)])
def test_shell_sum(image, length):
    result = shell_sum(image)

    assert len(result) == length


@pytest.mark.parametrize("clean_image", [camera(), newyork()])
def test_fsc(clean_image):
    clean_image = normalise(clean_image)
    noise1 = normal(size=clean_image.size).reshape(*clean_image.shape)
    noise2 = normal(size=clean_image.size).reshape(*clean_image.shape)

    image1 = clean_image + noise1
    image2 = clean_image + noise2

    correlations = fsc(image1, image2)

    assert sum(correlations[:10]) > sum(correlations[-40:-30])
