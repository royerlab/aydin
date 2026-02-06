"""Fourier Shell Correlation (FSC) for image quality assessment.

This module implements Fourier Shell Correlation, a frequency-domain measure
of similarity between two images. FSC is commonly used to estimate the
resolution of reconstructed images in microscopy and cryo-EM.

See: https://en.wikipedia.org/wiki/Fourier_shell_correlation
"""

import numpy
from numpy import fft


def fsc(image1, image2):
    """Calculate the Fourier Shell Correlation between two 2D images.

    Computes the normalized cross-correlation in concentric frequency
    shells between the Fourier transforms of the two input images.

    Parameters
    ----------
    image1 : numpy.typing.ArrayLike
        First input image (2D).
    image2 : numpy.typing.ArrayLike
        Second input image (2D), same shape as ``image1``.

    Returns
    -------
    fourier_shell_correlations : list of float
        FSC values for each frequency shell, from low to high frequency.
    """
    f_image1 = fft.fftshift(fft.fft2(image1))
    f_image2 = fft.fftshift(fft.fft2(image2))
    C = shell_sum(f_image1 * numpy.conjugate(f_image2))
    C = numpy.real(C)
    C1 = shell_sum(numpy.abs(f_image1) ** 2)
    C2 = shell_sum(numpy.abs(f_image2) ** 2)
    C = C.astype(numpy.float32)
    C1 = numpy.real(C1).astype(numpy.float32)
    C2 = numpy.real(C2).astype(numpy.float32)
    fourier_shell_correlations = abs(C) / numpy.sqrt(C1 * C2)

    return fourier_shell_correlations


def shell_sum(image):
    """Compute the sum of intensities over concentric shells centered on the image.

    For each radial distance from the image center, sums pixel values that
    fall within that shell (using both floor and ceiling rounding of the
    distance, averaged).

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        2D input image (typically a Fourier-domain image).

    Returns
    -------
    output : list of complex or float
        Sum of intensities for each concentric shell, ordered by radius.
    """
    len_x, len_y = image.shape
    r = numpy.arange(len_x) - numpy.floor(len_x / 2)
    c = numpy.arange(len_y) - numpy.floor(len_y / 2)
    [R, C] = numpy.meshgrid(r, c)
    map_floor = numpy.floor(numpy.sqrt(R**2 + C**2))
    map_ceil = numpy.ceil(numpy.sqrt(R**2 + C**2))

    nb_shells = int(numpy.max(map_ceil))

    indices_from_floor_map = [numpy.where(map_floor == i) for i in range(nb_shells)]
    indices_from_ceil_map = [numpy.where(map_ceil == i) for i in range(nb_shells)]

    output = [
        (sum(image[indices_from_floor_map[i]]) + sum(image[indices_from_ceil_map[i]]))
        / 2
        for i in range(nb_shells)
    ]

    return output
