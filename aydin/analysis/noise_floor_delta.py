import numpy

from aydin.analysis.fsc import fsc


def noise_floor_delta(noisy_array, denoised_array1, denoised_array2, axis: int) -> float:
    """
    This method calculates the noise floor level difference on spectra
    of a noisy and a denoised array (does NOT require groundtruth).

    Parameters
    ----------
    denoised_array : numpy.typing.ArrayLike
        Denoised nD array of interest
    noisy_array : numpy.typing.ArrayLike
        Noisy(original) nD array of interest
    axis : int
        Index of the axis to compute the noise floor delta along.
        One can pass -1 to do the computation on all axes.

    Returns
    -------
    float
        Calculated noise floor difference between the noisy and denoised arrays.

    """
    # TODO: handle the case where axis=-1
    # if axis == -1:
    #     numpy.newaxis(denoised_array, 0)

    correlations1 = fsc(noisy_array, denoised_array1)
    correlations2 = fsc(noisy_array, denoised_array2)

    halfbit1 = halfbit_curve(correlations1)
    halfbit2 = halfbit_curve(correlations2)

    return abs(halfbit1 - halfbit2)