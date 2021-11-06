import numpy
from numpy.linalg import norm
from scipy.fft import dct
from skimage.metrics import mean_squared_error


def spectral_psnr(norm_true_image, norm_test_image):
    """Spectral PSNR calculation

    Parameters
    ----------
    norm_true_image : numpy.typing.ArrayLike
    norm_test_image : numpy.typing.ArrayLike

    Returns
    -------
    Calculated PSNR : float

    Notes
    -----
    Interesting package: https://github.com/andrewekhalel/sewar
    """
    norm_true_image = norm_true_image / norm(norm_true_image.flatten(), 2)
    norm_test_image = norm_test_image / norm(norm_test_image.flatten(), 2)

    dct_norm_true_image = dct(
        dct(norm_true_image, axis=0, workers=-1), axis=1, workers=-1
    )
    dct_norm_test_image = dct(
        dct(norm_test_image, axis=0, workers=-1), axis=1, workers=-1
    )

    norm_dct_norm_true_image = dct_norm_true_image / norm(
        dct_norm_true_image.flatten(), 2
    )
    norm_dct_norm_test_image = dct_norm_test_image / norm(
        dct_norm_test_image.flatten(), 2
    )

    norm_true_image = abs(norm_dct_norm_true_image)
    norm_test_image = abs(norm_dct_norm_test_image)

    err = mean_squared_error(norm_true_image, norm_test_image)
    return 10 * numpy.log10(1 / err)


def spectral_mutual_information(image_a, image_b, normalised: bool = True):
    """Spectral mutual information

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
    image_b : numpy.typing.ArrayLike
    normalised : bool

    Returns
    -------
    mutual_information

    """
    norm_image_a = image_a / norm(image_a.flatten(), 2)
    norm_image_b = image_b / norm(image_b.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_image_a, axis=0, workers=-1), axis=1, workers=-1)
    dct_norm_test_image = dct(dct(norm_image_b, axis=0, workers=-1), axis=1, workers=-1)

    return mutual_information(
        dct_norm_true_image, dct_norm_test_image, normalised=normalised
    )


def joint_information(image_a, image_b, bins: int = 256):
    """Joint information

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
    image_b : numpy.typing.ArrayLike
    bins : int

    Returns
    -------
    joint information

    """
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    ji = joint_entropy_from_contingency(c_xy)
    return ji


def mutual_information(image_a, image_b, bins: int = 256, normalised: bool = True):
    """Mutual information

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
    image_b : numpy.typing.ArrayLike
    bins : int
    normalised : bool

    Returns
    -------
    mutual information

    """
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    mi = mutual_info_from_contingency(c_xy)
    mi = mi / joint_entropy_from_contingency(c_xy) if normalised else mi
    return mi


def joint_entropy_from_contingency(contingency):
    """Joint entropy from contingency

    Parameters
    ----------
    contingency : numpy.typing.ArrayLike

    Returns
    -------
    Joint entropy from contingency

    """

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # normalised contingency, i.e. probability:
    p = nz_val / contingency_sum

    # log contingency:
    log_p = numpy.log2(p)

    # Joint entropy:
    joint_entropy = -p * log_p

    return joint_entropy.sum()


def mutual_info_from_contingency(contingency):
    """Mutual info from contingency

    Parameters
    ----------
    contingency : numpy.typing.ArrayLike

    Returns
    -------
    Mutual info from contingency

    """

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # marginals:
    pi = numpy.ravel(contingency.sum(axis=1))
    pj = numpy.ravel(contingency.sum(axis=0))

    #
    log_contingency_nm = numpy.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(numpy.int64, copy=False) * pj.take(nzy).astype(
        numpy.int64, copy=False
    )
    log_outer = -numpy.log2(outer) + numpy.log2(pi.sum()) + numpy.log2(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - numpy.log2(contingency_sum))
        + contingency_nm * log_outer
    )
    return mi.sum()
