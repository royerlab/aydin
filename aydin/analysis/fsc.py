import numpy
from numpy import fft


def fsc(image1, image2):
    """
    Method that calculates Fourier Shell Correlations between
    the two provided 2D images. For more,
    https://en.wikipedia.org/wiki/Fourier_shell_correlation
    Parameters
    ----------
    image1 : numpy.ArrayLike
    image2 : numpy.ArrayLike
    Returns
    -------
    Sequence[float]
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
    """
    Method that calculates sum of intensities over image
    centric shells.
    Parameters
    ----------
    image : numpy.ArrayLike
    Returns
    -------
    Sequence[float]
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
