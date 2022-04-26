import numpy
from numpy.random import randint


def distill_noise_model(clean_array, noisy_array, nb_samples: int = 2 ** 18):
    """
    Given a clean array and a corresponding noisy array,
    this function analyses for each value in the clean array all the possible observed noisy values.
    A table is drawn that maps for each clean value the 'nb_samples' possible noisy values.
    This empirical noise model can be used to sample a noisy array from a clean one.

    Note: This function assumes unsigned int 16bit arrays.

    Parameters
    ----------
    clean_array : numpy.typing.ArrayLike
    noisy_array : numpy.typing.ArrayLike
    nb_samples : int

    Returns
    -------
    noise_model : numpy.ndarray

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
            print(f"... analysing values: [{clean_value}, {clean_value + 128}]")

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
    """Sample noise from model

    Parameters
    ----------
    clean_array : numpy.typing.ArrayLike
    noise_model : numpy.typing.ArrayLike

    Returns
    -------
    noisy : numpy.ndarray

    """

    shape = clean_array.shape

    x = clean_array.ravel()
    x[x < 0] = 0
    x = x.astype(numpy.uint16)

    rnd = randint(0, noise_model.shape[1], size=len(x))

    noisy = noise_model[x, rnd]

    noisy = noisy.reshape(*shape)

    return noisy
