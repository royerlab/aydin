"""Empirical noise model estimation and sampling.

This module provides functions to build an empirical noise model from paired
clean/noisy images, and to sample realistic noise from that model. The model
maps each possible clean pixel value to a distribution of observed noisy values.
"""

import numpy
from numpy.random import randint

from aydin.util.log.log import lprint


def distill_noise_model(clean_array, noisy_array, nb_samples: int = 2**18):
    """Build an empirical noise model from paired clean and noisy arrays.

    For each possible clean pixel value, collects all observed noisy values
    and stores a random sample of them. The resulting table can be used to
    generate realistic noise by looking up clean pixel values.

    Note: This function assumes unsigned 16-bit integer arrays.

    Parameters
    ----------
    clean_array : numpy.typing.ArrayLike
        The clean (ground truth) image array.
    noisy_array : numpy.typing.ArrayLike
        The corresponding noisy image array, same shape as ``clean_array``.
    nb_samples : int
        Number of noisy value samples to store per clean value.

    Returns
    -------
    noise_model : numpy.ndarray
        Array of shape ``(max_clean_value + 1, nb_samples)`` with dtype
        uint16. Each row contains sampled noisy values for that clean value.
    """

    x = clean_array.ravel()
    y = noisy_array.ravel()

    x[x < 0] = 0
    y[y < 0] = 0

    x = x.astype(numpy.uint16)
    y = y.astype(numpy.uint16)

    # TODO: remove the following noqa when you can
    clean_min, clean_max = x.min(), x.max()  # noqa: F841

    noise_model = numpy.zeros(shape=(clean_max + 1, nb_samples), dtype=numpy.uint16)

    last_indices = numpy.array([0])
    for clean_value in range(0, clean_max + 1):

        if clean_value % 128 == 0:
            lprint(f"... analysing values: [{clean_value}, {clean_value + 128}]")

        (indices,) = numpy.where(x == clean_value)

        if indices.size == 0:
            indices = last_indices

        noisy_values = y[indices]

        noise_model[clean_value] = noisy_values[
            randint(0, max(1, len(noisy_values)), size=nb_samples)
        ]

        last_indices = indices

    return noise_model


def sample_noise_from_model(clean_array, noise_model):
    """Generate a noisy image by sampling from an empirical noise model.

    Each pixel in the clean array is used as an index into the noise model
    to look up a random noisy value.

    Parameters
    ----------
    clean_array : numpy.typing.ArrayLike
        The clean image array to add noise to. Values are clipped to
        non-negative and cast to uint16.
    noise_model : numpy.typing.ArrayLike
        Noise model array as returned by ``distill_noise_model``.

    Returns
    -------
    noisy : numpy.ndarray
        Noisy image with the same shape as ``clean_array``, dtype uint16.
    """

    shape = clean_array.shape

    x = clean_array.ravel()
    x[x < 0] = 0
    x = x.astype(numpy.uint16)

    rnd = randint(0, noise_model.shape[1], size=len(x))

    noisy = noise_model[x, rnd]

    noisy = noisy.reshape(*shape)

    return noisy
