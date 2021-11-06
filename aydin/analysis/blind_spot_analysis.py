from typing import Tuple, List

import numpy
import scipy
from numpy import argmax, unravel_index
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter

from aydin.util.crop.rep_crop import representative_crop


def auto_detect_blindspots(
    image,
    batch_axes: Tuple[bool] = None,
    channel_axes: Tuple[bool] = None,
    threshold=0.01,
    max_blind_spots=3,
    max_range: int = 3,
    window: int = 31,
    crop_border: int = 2,
    fastmode: bool = True,
) -> Tuple[List[Tuple[int, ...]], numpy.ndarray]:
    """Automatically determines the list of blind-spots for Noise2Self

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Image for which to conduct blind-spot analysis

    batch_axes : Tuple[bool]
        Batch axes for image

    channel_axes : Tuple[bool]
        Channel axes for image

    threshold
        threshold for inclusion as a blind-spot

    max_blind_spots
        Max number of blindspots

    max_range : int
        maximal range of the returned blind-spots and noise autocorrelogram

    window : int
        window used for computing autocorrelogram (best left unchanged)

    crop_border : int
        Cropping border

    fastmode : str
        When True we use an even faster approach.


    Returns
    -------
    Tuple of list of blindspots and the noise autocorrelogram

    """

    # Handle default values for batch and channel dim specification:
    if batch_axes is None:
        batch_axes = (False,) * image.ndim
    if channel_axes is None:
        channel_axes = (False,) * image.ndim

    # Ensure there is at least one batch or channel dimensions:
    if batch_axes == (False,) * image.ndim and channel_axes == (False,) * image.ndim:
        image = image[numpy.newaxis, ...]
        batch_axes = (True,) + batch_axes
        channel_axes = (False,) + channel_axes

    # we collapse all batch and channel dimensions together:
    shape = tuple(
        s for s, b, c in zip(image.shape, batch_axes, channel_axes) if not b and not c
    )
    if len(shape) < image.ndim:
        shape = (-1,) + shape
    image = numpy.reshape(image, shape)

    # And pick the image with the most variance:
    chosen_image = None
    chosen_image_variance = -1
    for image_index in range(image.shape[0]):
        one_image = image[image_index]
        variance = numpy.var(one_image)
        if variance > chosen_image_variance and not numpy.isnan(variance):
            chosen_image_variance = variance
            chosen_image = one_image

    # chosen_image should not be None, but if does happen (happened once!) then let's play safe:
    if chosen_image is not None:
        image = chosen_image

    # First we need to remove the borders of the image, as sometimes the borders have artefacts:
    if crop_border > 0 and all(s > 2 * crop_border for s in image.shape):
        crop_slice = tuple(
            slice(max(s // 16, crop_border), -max(s // 16, crop_border))
            for s in image.shape
        )
        image = image[crop_slice]

    if fastmode:
        # obtain representative crop, to speed things up...
        image = representative_crop(image, crop_size=int(1e6), favour_odd_lengths=True)

    # We compute the autocorrelogram of the noise:
    noise_auto = noise_autocorrelation(image, max_range=max_range, window=window)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(noise_auto, name='noise_auto')

    # What is the intensity of the nth strongest correlation?
    noise_auto_flat = noise_auto.flatten()
    noise_auto_flat.sort()
    nth_strongest_correlation = noise_auto_flat[-max_blind_spots]

    # We adjust the threshold to take into account the max number of blindspots requested:
    threshold = max(threshold, nth_strongest_correlation)

    # We list the blind spots:
    blind_spots = []
    for idx, x in numpy.ndenumerate(noise_auto):
        if x >= threshold:
            blind_spot = tuple(n - max_range for n in idx)
            blind_spots.append(blind_spot)

    # Remove any constant offset per dimension:
    blind_spots = numpy.array(blind_spots)
    blind_spots = blind_spots - numpy.mean(blind_spots, axis=0, keepdims=True).astype(
        dtype=numpy.int32
    )

    # Convert back to list of tuples:
    blind_spots = list(tuple(a) for a in blind_spots)

    return blind_spots, noise_auto


def noise_autocorrelation(image, max_range: int = 3, window: int = 31) -> numpy.ndarray:
    """This function computes the noise autocorrelogram.

    Principle: We simply divide the autocorrelogram of the raw image by the autocorrelogram of a naively denoised image.
    What is left is the autocorrelogram of the noise.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        image to compute the noise autocorrelation
    max_range : int
        expected maximal range of correlation (3 is a good number!)
    window : int
        window size for correlation computation (31 is a good number!)

    Returns
    -------
    analysis : numpy.ndarray
        noise autocorrelogram of shape (max_range*2+1,)*ndim  where ndim is the number of dimensions of the input image.
    """

    # Enfor

    # First we compute the auto-correlation of the raw image:
    auto_corr = _autocorrelation(image, window=window)

    # Second we compute the auto-correlation of the image after 'rough' denoising:
    blurred_image = gaussian_filter(image, sigma=0.5)
    # blurred_image = median_filter(blurred_image, size=2)
    blurred_auto_corr = _autocorrelation(blurred_image, window=window)

    # We 'remove' by division the autocorrelogram fixed_pattern that is common to both:
    analysis = auto_corr / blurred_auto_corr

    # Now we compute the maximum intensity outside of the central region:
    outside = analysis.copy()
    # center = tuple((min(s, window) - 1) // 2 for s in analysis.shape)
    center = unravel_index(argmax(analysis), analysis.shape)

    # Max range might be too much for very shallow dimensions, and for non odd dimensions the center
    # is not at the cente so we need to reduce the range accordingly:
    range = tuple(min(c, s - c, max_range) for c, s in zip(center, analysis.shape))

    # Let's compute the center slice:
    center_slice = tuple(slice(c - r, c + r + 1, 1) for c, r in zip(center, range))
    outside[center_slice] = 0
    floor = numpy.max(outside)

    # And remove it:
    analysis = analysis - floor
    analysis = analysis.clip(0, numpy.math.inf)

    # We normalise to a sum of 1:
    analysis /= analysis.max()

    # Now we have the noise autocorelogram, we can crop that to the expected max range:
    analysis = analysis[center_slice]

    return analysis


def _autocorrelation(image, window: int = 31) -> numpy.ndarray:
    """Computes the autocorrelation of an image over a cropped window around the origin

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        image
    window : int
        window size

    Returns
    -------
    array : numpy.ndarray

    """
    image = image.astype(numpy.float32)
    image /= norm(image)

    array = _phase_correlation(image, image)
    shift = tuple(min(window, s) // 2 for s in image.shape)
    array = numpy.roll(array, shift=shift, axis=range(image.ndim))
    slice_tuple = (slice(0, window),) * image.ndim
    array = array[slice_tuple]
    return array


def _phase_correlation(image, reference_image) -> numpy.ndarray:
    """Computes the phase correlation between am image and a reference image.
    Note: both images must be of the same shape.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
    reference_image : numpy.ArrayLike

    Returns
    -------
    r : numpy.ndarray

    """
    G_a = scipy.fft.fftn(image, workers=-1)
    G_b = scipy.fft.fftn(reference_image, workers=-1)
    conj_b = numpy.ma.conjugate(G_b)
    R = G_a * conj_b
    r = numpy.absolute(scipy.fft.ifftn(R, workers=-1))
    return r
