"""Image quality metrics for denoising evaluation.

This module provides functions for measuring image quality including SSIM, PSNR,
spectral PSNR, mutual information, and joint entropy. These metrics are used
to evaluate the performance of denoising algorithms.
"""

import numpy
from numpy.linalg import norm
from scipy.fft import dct
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as _ssim

from aydin.util.log.log import lprint


def ssim(im1, im2, **kwargs):
    """Compute the Structural Similarity Index (SSIM) between two images.

    Wrapper around ``skimage.metrics.structural_similarity`` that automatically
    sets ``data_range`` for floating-point images and converts the deprecated
    ``multichannel`` parameter to ``channel_axis``.

    Parameters
    ----------
    im1 : numpy.typing.ArrayLike
        First input image.
    im2 : numpy.typing.ArrayLike
        Second input image, same shape as ``im1``.
    **kwargs
        Additional keyword arguments passed to ``structural_similarity``.

    Returns
    -------
    ssim_value : float
        SSIM value between the two images.
    """
    # If data_range not specified and images are floating point, set data_range
    if 'data_range' not in kwargs:
        if numpy.issubdtype(im1.dtype, numpy.floating) or numpy.issubdtype(
            im2.dtype, numpy.floating
        ):
            kwargs['data_range'] = max(im1.max() - im1.min(), im2.max() - im2.min())
    # Convert deprecated multichannel to channel_axis
    if 'multichannel' in kwargs:
        multichannel = kwargs.pop('multichannel')
        if multichannel:
            kwargs['channel_axis'] = -1
    return _ssim(im1, im2, **kwargs)


def calculate_print_psnr_ssim(clean_image, noisy_image, denoised_image):
    """Calculate and print PSNR and SSIM for noisy and denoised images.

    Computes PSNR and SSIM between the clean image and both the noisy
    and denoised versions, then prints the results to stdout.

    Parameters
    ----------
    clean_image : numpy.typing.ArrayLike
        Ground truth clean image.
    noisy_image : numpy.typing.ArrayLike
        Noisy version of the image.
    denoised_image : numpy.typing.ArrayLike
        Denoised version of the image.

    Returns
    -------
    psnr_noisy : float
        PSNR between the clean and noisy image.
    psnr_denoised : float
        PSNR between the clean and denoised image.
    ssim_noisy : float
        SSIM between the clean and noisy image.
    ssim_denoised : float
        SSIM between the clean and denoised image.
    """
    psnr_noisy = psnr(clean_image, noisy_image)
    ssim_noisy = ssim(clean_image, noisy_image)
    psnr_denoised = psnr(clean_image, denoised_image)
    ssim_denoised = ssim(clean_image, denoised_image)
    lprint("noisy   :", psnr_noisy, ssim_noisy)
    lprint("denoised:", psnr_denoised, ssim_denoised)

    return psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised


def spectral_psnr(norm_true_image, norm_test_image):
    """Compute the spectral Peak Signal-to-Noise Ratio (PSNR) in DCT domain.

    Normalizes both images, transforms them to the DCT domain, and
    computes the PSNR between their spectral representations.

    Parameters
    ----------
    norm_true_image : numpy.typing.ArrayLike
        Ground truth image.
    norm_test_image : numpy.typing.ArrayLike
        Test image to compare against the ground truth.

    Returns
    -------
    spectral_psnr : float
        Spectral PSNR value in dB.

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
    """Compute spectral mutual information between two images in DCT domain.

    Normalizes both images, transforms them to the DCT domain, and
    computes their mutual information.

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
        First input image.
    image_b : numpy.typing.ArrayLike
        Second input image.
    normalised : bool
        If True, normalizes the mutual information by the joint entropy.

    Returns
    -------
    mi : float
        Spectral mutual information value.
    """
    norm_image_a = image_a / norm(image_a.flatten(), 2)
    norm_image_b = image_b / norm(image_b.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_image_a, axis=0, workers=-1), axis=1, workers=-1)
    dct_norm_test_image = dct(dct(norm_image_b, axis=0, workers=-1), axis=1, workers=-1)

    return mutual_information(
        dct_norm_true_image, dct_norm_test_image, normalised=normalised
    )


def joint_information(image_a, image_b, bins: int = 256):
    """Compute the joint information (joint entropy) of two images.

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
        First input image.
    image_b : numpy.typing.ArrayLike
        Second input image.
    bins : int
        Number of histogram bins for the joint distribution.

    Returns
    -------
    ji : float
        Joint entropy value in bits.
    """
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    ji = joint_entropy_from_contingency(c_xy)
    return ji


def mutual_information(image_a, image_b, bins: int = 256, normalised: bool = True):
    """Compute the mutual information between two images.

    Parameters
    ----------
    image_a : numpy.typing.ArrayLike
        First input image.
    image_b : numpy.typing.ArrayLike
        Second input image.
    bins : int
        Number of histogram bins for the joint distribution.
    normalised : bool
        If True, normalizes by the joint entropy (returns a value in [0, 1]).

    Returns
    -------
    mi : float
        Mutual information value (optionally normalized).
    """
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    mi = mutual_info_from_contingency(c_xy)
    mi = mi / joint_entropy_from_contingency(c_xy) if normalised else mi
    return mi


def joint_entropy_from_contingency(contingency):
    """Compute joint entropy from a contingency table.

    Parameters
    ----------
    contingency : numpy.typing.ArrayLike
        2D contingency table (joint histogram) of two variables.

    Returns
    -------
    joint_entropy : float
        Joint entropy in bits (log base 2).
    """

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingency table:
    contingency_sum = contingency.sum()

    # normalised contingency, i.e. probability:
    p = nz_val / contingency_sum

    # log contingency:
    log_p = numpy.log2(p)

    # Joint entropy:
    joint_entropy = -p * log_p

    return joint_entropy.sum()


def mutual_info_from_contingency(contingency):
    """Compute mutual information from a contingency table.

    Parameters
    ----------
    contingency : numpy.typing.ArrayLike
        2D contingency table (joint histogram) of two variables.

    Returns
    -------
    mi : float
        Mutual information in bits (log base 2).
    """

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingency table:
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
