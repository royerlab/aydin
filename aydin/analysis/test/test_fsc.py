# import pytest

from aydin.analysis.fsc import shell_sum, fsc


# @pytest.mark.parametrize("image", [2, 3, 5, 8])
def test_shell_sum(image, length):
    result = shell_sum(image)

    assert len(result) == length


def test_fsc(image1, image2):
    correlations = fsc(image1, image2)

    assert correlations.shape == image1.shape == image2.shape
