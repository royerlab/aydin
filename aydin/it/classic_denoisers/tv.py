from functools import partial
from typing import Optional

import numpy
from numpy.typing import ArrayLike
from skimage.restoration import denoise_tv_bregman
from skimage.restoration._denoise import _denoise_tv_chambolle_nd

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_tv(
    image: ArrayLike,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_normal,
    optimiser: str = _defaults.default_optimiser,
    max_num_evaluations: int = _defaults.default_max_evals_normal,
    enable_extended_blind_spot: bool = True,
    display_images: bool = False,
    display_crop: bool = False,
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

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        (advanced)

    enable_extended_blind_spot: bool
        Set to True to enable extended blind-spot detection.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced)

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
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

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
        calibrate_denoiser(
            crop,
            _denoise_tv,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            blind_spots=enable_extended_blind_spot,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = image.nbytes * 2  # gradient image

    return denoise_tv, best_parameters, memory_needed


def denoise_tv(
    image: ArrayLike, algorithm: str = 'bregman', weight: float = 1, **kwargs
):
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
