import math

import numpy
from numpy.linalg import norm
from scipy.fft import dctn

from aydin.analysis.resolution_estimate import resolution_estimate


def snr_estimate(image) -> float:
    """Estimates the signal to noise ratio of an image in DB.

    A value of 0 means that the signal and noise have roughly the same energy,
    a negative value means that the noise is stronger than the signal,
    and reciprocally, a positive value means that the signal is stronger
    than the noise.

    Parameters
    ----------
    image : numpy.typing.ArrayLike

    Returns
    -------
    Returns an estimate of the image's signal-to-noise ratio in dB.
    """

    # First we estimate resolution:
    frequency, image = resolution_estimate(image)

    # Compute the DCT:
    image_dct = dctn(image, workers=-1)

    # Compute frequency map:
    f = numpy.zeros_like(image)
    axis_grid = tuple(numpy.linspace(0, 1, s) for s in image.shape)
    for x in numpy.meshgrid(*axis_grid, indexing='ij'):
        f += x ** 2
    f = numpy.sqrt(f)

    # define two domains:
    signal_domain = f <= frequency
    noise_domain = f > frequency

    # First we measure the energy of both signa and noise:
    signal_energy = norm(image_dct[signal_domain]) ** 2
    noise_energy = norm(image_dct[noise_domain]) ** 2

    # However, this is an underestimate of the noise, because we assume that
    # the noise is uniformly distributed in frequency space. Therefore, we need
    # to correct this first estimate. For this we need the 'volume, in frequency
    # space of both domains:

    signal_domain_volume = numpy.sum(signal_domain) / image_dct.size
    noise_domain_volume = 1 - signal_domain_volume

    # We re-estimate the noise energy assuming the same density over the whole
    # spectrum:
    corrected_noise_energy = noise_energy / noise_domain_volume

    # For the signal energy we remove the energy that comes from the noise:
    corrected_signal_energy = (
        signal_energy - signal_domain_volume * corrected_noise_energy
    )

    # We can't let the signal energy to go below zero:
    corrected_signal_energy = max(1e-16, corrected_signal_energy)

    # Signal to noise ratio in dB:
    noise_ratio = 10 * math.log10(corrected_signal_energy / corrected_noise_energy)

    return noise_ratio
