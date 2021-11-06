from typing import Sequence
import numpy


def extend_nd(available_dims: Sequence[int] = (2,)):
    """
    Decorator that extends to nD a denoising function that is limited to images
    of a few dimensions (e.g. 2 or 3).

    Parameters
    ----------
    available_dims: Tuple[int]

    Returns
    -------

    """

    # Actual decorator that varies given the parameters
    def decorator(function):
        def wrapper(*args, **kwargs):

            # The first argument must be the image:
            image = args[0]

            # The dimension of the image:
            ndim = image.ndim

            if ndim in available_dims:
                # If we can denoise directly, let's do it:
                return function(*args, **kwargs)
            else:

                # What are the smallest available image dimensions that we can denoise?
                smallest_dim = min(available_dims)

                if ndim < smallest_dim:
                    # In this case we are trying to denoise an image with less dimensions that what the function can do,
                    # So we need to add extra dimensions:

                    extended_image = image
                    for _ in range(smallest_dim - ndim):
                        extended_image = extended_image[numpy.newaxis, ...]

                    # we apply the denoising function:
                    denoised = function(extended_image, *args[1:], **kwargs)

                else:
                    # In this case we try to denoise an image with more dimensions that what the function can do,
                    # so we have to denoise hyperplane by hyperplane.

                    # In this case, this is the largest available dimension below the image dimension:
                    largest_dim = max((n for n in available_dims if n < ndim))

                    # Let's use order the image by increasing dimension length,
                    # this way we will denoise the largest sub images first:
                    # The corresponding permutation is:
                    permutation = tuple(numpy.argsort(image.shape))

                    # let's permutate the array dimensions accordingly:
                    reshaped_image = numpy.transpose(image, axes=permutation)

                    # Let's collapse the first ndim-largest_dim dimensions:
                    collapsed_image = numpy.reshape(
                        reshaped_image,
                        newshape=(-1,) + reshaped_image.shape[ndim - largest_dim :],
                    )

                    # We denoise each such image:
                    collapsed_denoised = numpy.stack(
                        tuple(function(i, *args[1:], **kwargs) for i in collapsed_image)
                    )

                    # Give it back its original shape:
                    reshaped_denoised = numpy.reshape(
                        collapsed_denoised, newshape=reshaped_image.shape
                    )
                    denoised = numpy.transpose(
                        reshaped_denoised, axes=tuple(numpy.argsort(permutation))
                    )

                return denoised

        return wrapper

    return decorator
