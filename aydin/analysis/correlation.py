import random

import numpy
import scipy


def correlation_distance(input_image, target_image=None, method: str = 'firstmin'):
    """Computes correlation distances

    Parameters
    ----------
    input_image : numpy.typing.ArrayLike
    target_image : numpy.typing.ArrayLike
    method : str

    Returns
    -------
    Tuple of correlation_distances : tuple

    """

    correlation_curves_list = correlation(input_image, target_image)

    correlation_distances_list = []

    for correlation_curve in correlation_curves_list:

        if correlation_curve is None:
            correlation_distances_list.append(0)
        else:
            length = correlation_curve.shape[0]

            if method == 'zerocross':
                correlation_distance = 0
                for distance in range(length):
                    value = correlation_curve[distance]
                    if value < 0:
                        correlation_distance = distance
                        break

                correlation_distances_list.append(correlation_distance)

            elif method == 'firstmin':
                last_value = 1
                correlation_distance = 0
                for distance in range(length):
                    value = correlation_curve[distance]
                    if value > last_value and value < 0:
                        correlation_distance = distance - 1
                        break
                    last_value = value

                correlation_distances_list.append(correlation_distance)

            elif method == 'min':
                min_value = 1
                min_distance = 0
                for distance in range(length):
                    value = correlation_curve[distance]
                    if value < min_value and value < 0:
                        min_value = value
                        min_distance = distance
                correlation_distances_list.append(min_distance)

    return tuple(correlation_distances_list)


def correlation(
    input_image,
    target_image=None,
    nb_samples: int = 4 * 1024,
    max_length: int = 256,
    smooth: bool = True,
):
    """Computes correlation

    Parameters
    ----------
    input_image : numpy.typing.ArrayLike
    target_image : numpy.typing.ArrayLike
    nb_samples : int
    max_length : int
    smooth : bool

    Returns
    -------
    Tuple of correlations : tuple

    """

    # Determine image(s)  shape:
    shape = input_image.shape

    # Initialise target image if  None:
    if target_image is None:
        target_image = input_image

    # Makes sure that the images have the same shape:
    if input_image is not target_image and input_image.shape != target_image.shape:
        raise ValueError('Input image and target image has different shapes.')

    # Number of dimensions:
    nb_dim = len(shape)

    # This list will contain the correlation vectors for each and every dimension:
    corr_list = []

    # We iterate for each dimension:
    for dim in range(nb_dim):

        dim_length = shape[dim]

        if dim_length >= 3:

            max_length_dim = min(dim_length, max_length)

            corr_samples_list = []
            counter = 0

            for sample in range(nb_samples):
                slice_list = list(random.randrange(0, shape[i]) for i in range(nb_dim))

                pos = random.randrange(0, 1 + shape[dim] - max_length_dim)
                slice_list[dim] = slice(pos, pos + max_length_dim, 1)

                line_array_input = input_image[tuple(slice_list)]
                line_array_target = target_image[tuple(slice_list)]

                line_array_input = line_array_input.astype(numpy.float, copy=False)
                line_array_target = line_array_target.astype(numpy.float, copy=False)

                line_array_input = line_array_input - (
                    line_array_input.sum() / line_array_input.shape[0]
                )
                line_array_target = line_array_target - (
                    line_array_target.sum() / line_array_target.shape[0]
                )

                corr = numpy.correlate(line_array_input, line_array_target, mode='full')
                corr = corr[corr.size // 2 :]

                if corr[0] <= 0:
                    continue

                # corr = numpy.abs(corr)

                corr_samples_list.append(corr)

                counter += 1

            if len(corr_samples_list) > 0:
                corr_samples_stack = numpy.stack(corr_samples_list)
                corr_avg = numpy.median(corr_samples_stack, axis=0)

                # corr_avg = corr_avg / numpy.sum(corr_avg)

                if smooth and corr_avg.size >= 3:
                    corr_avg[1:] = numpy.convolve(
                        corr_avg, numpy.ones(3) / 3.0, mode='same'
                    )[1:]
                    corr_avg[1:] = numpy.convolve(
                        corr_avg, numpy.ones(3) / 3.0, mode='same'
                    )[1:]
                    corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=3)[1:]
                    corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=5)[1:]
                    corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=7)[1:]
                    # corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
                    # corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
                    # corr_avg = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')

                corr_avg = corr_avg / corr_avg[0]
            else:
                corr_avg = None

        else:
            # Dimension is way too short:
            corr_avg = None

        corr_list.append(corr_avg)

    return tuple(corr_list)
