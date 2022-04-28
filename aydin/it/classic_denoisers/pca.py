import math
from functools import partial
from math import prod
from typing import Optional, Union, Tuple, List

import numpy
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser
from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import (
    extract_patches_nd,
    reconstruct_from_nd_patches,
)


def calibrate_denoise_pca(
    image: ArrayLike,
    patch_size: Optional[Union[int, Tuple[int], str]] = None,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_normal.value,
    optimiser: str = _defaults.default_optimiser.value,
    max_num_evaluations: int = _defaults.default_max_evals_hyperlow.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = _defaults.default_jinv_interpolation_mode.value,
    multi_core: bool = True,
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Principal Component Analysis (PCA) denoiser for the given
    image and returns the optimal parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate spectral denoiser for.

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be an int s that corresponds to isotropic patches of shape: (s,)*image.ndim,
        or a tuple of ints. By default (None) the patch size is chosen automatically
        to give the best results.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        Increase this number by factors of two if denoising quality is
        unsatisfactory -- this can be important for very noisy images.
        Values to try are: 65000, 128000, 256000, 320000.
        We do not recommend values higher than 512000.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    blind_spots: bool
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    jinv_interpolation_mode: str
        J-invariance interpolation mode for masking. Can be: 'median' or
        'gaussian'.
        (advanced)

    multi_core: bool
        Use all CPU cores during calibration.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced) (hidden)

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced) (hidden)

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

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # Ranges:
    threshold_range = list(numpy.linspace(0, 1, max_num_evaluations).tolist())

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'threshold': threshold_range}

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'patch_size': patch_size}

    # Partial function:
    _denoise_pca = partial(
        denoise_pca, **(other_fixed_parameters | {'multi_core': multi_core})
    )

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_pca,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes + 6 * image.nbytes * math.prod(patch_size)

    return denoise_pca, best_parameters, memory_needed


def denoise_pca(
    image: ArrayLike,
    patch_size: Optional[Union[int, Tuple[int]]] = None,
    threshold: float = 0.1,
    reconstruction_gamma: float = 0,
    multi_core: bool = True,
):
    """
    Denoises the given image by first applying the patch
    transform, and then doing a PCA projection of these patches
    along the first components. The cut-off is set by a
    threshold parameter.

    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be: 'full' for a single patch covering the whole image, 'half', 'quarter',
        or an int s that corresponds to isotropic patches of shape: (s,)*image.ndim,
        or a tuple of ints. By default (None) the patch size is chosen automatically
        to give the best results.

    threshold: float
        Threshold for proportion of components to retain. Between 0 and 1

    reconstruction_gamma: float
        Patch reconstruction parameter

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery.


    Returns
    -------
    Denoised image

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # First we apply the patch transform:
    patches = extract_patches_nd(image, patch_size=patch_size)

    # reshape patches as vectors:
    original_patches_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    # PCA dim reduction setup:
    n_components = 1 + max(0, int(threshold * (prod(patch_size) - 1)))
    pca = PCA(n_components=n_components)

    # Project patches:
    patches = pca.inverse_transform(pca.fit_transform(patches))

    # reshape patches back to their original shape:
    patches = patches.reshape(original_patches_shape)

    # Transform back from patches to image:
    denoised_image = reconstruct_from_nd_patches(
        patches, image.shape, gamma=reconstruction_gamma
    )

    # Cast back to float32 if needed:
    denoised_image = denoised_image.astype(numpy.float32, copy=False)

    return denoised_image
