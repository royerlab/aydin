"""Loss functions for J-invariance-based denoiser calibration.

Provides loss functions used to evaluate denoising quality in a
self-supervised manner, including MSE, MAE, L-half, and SSIM-based losses.
"""

import numpy
from numba import jit
from skimage.metrics import structural_similarity


@jit(nopython=True, parallel=True)
def mean_squared_error(image0, image1):
    """Compute the mean squared error between two images.

    Parameters
    ----------
    image0 : numpy.ndarray
        First image (typically the original noisy image at masked positions).
    image1 : numpy.ndarray
        Second image (typically the denoised image at masked positions).

    Returns
    -------
    float
        Mean squared error value.
    """
    return numpy.mean((image0 - image1) ** 2)


@jit(nopython=True, parallel=True)
def mean_absolute_error(image_a, image_b):
    """Compute the mean absolute error (L1 loss) between two images.

    Parameters
    ----------
    image_a : numpy.ndarray
        First image.
    image_b : numpy.ndarray
        Second image.

    Returns
    -------
    float
        Mean absolute error value.
    """
    return numpy.mean(numpy.absolute(image_a - image_b))


@jit(nopython=True, parallel=True)
def lhalf_error(image_a, image_b):
    """Compute the L-half pseudo-norm error between two images.

    Computes mean(|a - b|^0.5)^2, which is more robust to outliers
    than L1 or L2 losses.

    Parameters
    ----------
    image_a : numpy.ndarray
        First image.
    image_b : numpy.ndarray
        Second image.

    Returns
    -------
    float
        L-half error value.
    """
    return numpy.mean(numpy.absolute(image_a - image_b) ** 0.5) ** 2


def structural_loss(image_a, image_b):
    """Compute structural dissimilarity loss (1 - SSIM) between two images.

    Parameters
    ----------
    image_a : numpy.ndarray
        First image.
    image_b : numpy.ndarray
        Second image.

    Returns
    -------
    numpy.float32
        Structural dissimilarity value in [0, 2].
    """
    return numpy.asarray(1 - structural_similarity(image_a, image_b)).astype(
        numpy.float32
    )
