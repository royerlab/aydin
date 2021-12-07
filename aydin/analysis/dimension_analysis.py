import numpy

from aydin.it.classic_denoisers.gaussian import calibrate_denoise_gaussian
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import lprint


def dimension_analysis_on_image(image, max_length=256, crop_size_in_voxels=1000000):

    # obtain representative crop, to speed things up...
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, max_time_in_seconds=0.1
    )

    crop = crop.astype(dtype=numpy.float32)

    # Number of dimensions:
    nb_dim = len(crop.shape)

    axis_sigma = []

    # We iterate for each dimension:
    for axis in range(nb_dim):

        projection = numpy.max(crop, axis=tuple(a for a in range(nb_dim) if a != axis))

        _, best_parameters, _ = calibrate_denoise_gaussian(
            projection, max_num_truncate=None, max_num_evaluations=64
        )

        sigma = best_parameters['sigma']

        lprint(f"sigma={sigma}")

        axis_sigma.append(sigma)

    lprint(f"sigma={axis_sigma}")
