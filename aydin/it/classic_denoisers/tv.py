from functools import partial
from typing import Optional

import numpy
from skimage.restoration import denoise_tv_bregman
from skimage.restoration._denoise import _denoise_tv_chambolle_nd

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart


def calibrate_denoise_tv(
    image,
    crop_size_in_voxels: Optional[int] = 64000,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Total Variation (TV) denoiser for the given image and
    returns the optimal parameters obtained using the N2S loss.

    Note: we use the scikt-image implementation of TV denoising.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate TV denoiser for.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

    other_fixed_parameters: dict
        Any other fixed parameters

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.

    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # Sigma spatial range:
    range = max(1e-10, numpy.max(image) - numpy.min(image))
    # weight_range = np.arange(0.01, 1.5*std, 0.05*std) ** 1.5
    weight_range = (0.01 * range, 1.5 * range)

    # Algorithms:
    algorithms = ['bregman', 'chambolle'] if image.ndim <= 2 else ['chambolle']
    # Note: scikitimage implementation of bregman does not support more than 2D

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'weight': weight_range,
        'isotropic': [False, True],
        'algorithm': algorithms,
        # 'algorithm': ['chambolle','bregman'], #
    }

    # Partial function:
    _denoise_tv = partial(denoise_tv, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_smart(
            crop,
            _denoise_tv,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = image.nbytes * 2  # gradient image

    return denoise_tv, best_parameters, memory_needed


def denoise_tv(image, algorithm: str = 'bregman', weight: float = 1, **kwargs):
    """
    Denoises the given image using either scikit-image
    implementation of Bregman or Chambolle <a
    href="https://en.wikipedia.org/wiki/Total_variation_denoising">Total
    Variation (TV) denoising</a>. We attempt to rescale the weight parameter to
    obtain similar results between Bregman and Chambolle for the
    same weight.
    \n\n
    Note: Because of limitations of the current scikit-image
    implementation, if an image with more than 2 dimensions
    is passed, we use the Chambolle implementation as it
    supports nD images...

    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    algorithm: str
        Algorithm to apply: either 'bregman' or 'chambolle'

    weight: float
        Weight to balance data term versus prior term. Larger values correspond
        to a stronger prior during optimisation, and thus more aggressive
        denoising

    kwargs
        Any other parameters to be passed to scikit-image implementations

    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # algorithm = parameters.pop('weight')
    if algorithm == 'bregman' and image.ndim <= 2:
        return denoise_tv_bregman(image, weight=weight * 10, **kwargs)
    if algorithm == 'chambolle' or image.ndim > 2:
        if 'isotropic' in kwargs:
            kwargs.pop('isotropic')
        return _denoise_tv_chambolle_nd(image, weight=weight, **kwargs)
