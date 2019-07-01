import random
import numpy
import scipy


def correlation(image, nb_samples=1024, max_length=256, smooth=True):

    shape = image.shape
    nb_dim = len(shape)

    corr_list = []

    for dim in range(nb_dim):

        max_length_dim = min(shape[dim],max_length)

        corr_samples_list = []
        counter = 0

        for sample in range(nb_samples):
            slice_list = list( random.randrange(0, shape[i]) for i in range(nb_dim) )

            pos = random.randrange(0, 1+shape[dim]-max_length_dim)
            slice_list[dim] = slice(pos,pos+max_length_dim,1)

            line_array = image[tuple(slice_list)]

            line_array = line_array.astype(numpy.float)

            line_array = line_array - (line_array.sum()/line_array.shape[0])

            corr = numpy.correlate(line_array, line_array, mode='full')
            corr = corr[corr.size // 2:]

            if corr[0]<=0:
                continue

            #corr = numpy.abs(corr)

            corr_samples_list.append(corr)

            counter+=1

        corr_samples_stack = numpy.stack(corr_samples_list)
        corr_avg           = numpy.median(corr_samples_stack, axis=0)

        #corr_avg = corr_avg / numpy.sum(corr_avg)

        if smooth:
            corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
            corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
            corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=3)[1:]
            corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=5)[1:]
            corr_avg[1:] = scipy.signal.medfilt(corr_avg, kernel_size=7)[1:]
            #corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
            #corr_avg[1:] = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')[1:]
            #corr_avg = numpy.convolve(corr_avg, numpy.ones(3) / 3.0, mode='same')

        corr_avg = corr_avg / corr_avg[0]

        corr_list.append(corr_avg)

    return tuple(corr_list)


def correlation_distance(image, method='firstmin'):
    correlation_curves_list = correlation(image)

    correlation_distances_list = []

    for correlation_curve in correlation_curves_list:

        length = correlation_curve.shape[0]

        if method == 'zerocross':
            for distance in range(length):
                value = correlation_curve[distance]
                if value<0:
                    correlation_distances_list.append(distance)
                    break
        elif method=='firstmin':
            last_value = 1
            for distance in range(length):
                value = correlation_curve[distance]
                if value>last_value and value<0:
                    correlation_distances_list.append(distance-1)
                    break
                last_value = value
        elif method=='min':
            min_value = 1
            min_distance = 0
            for distance in range(length):
                value = correlation_curve[distance]
                if value<min_value and value<0:
                    min_value = value
                    min_distance = distance
            correlation_distances_list.append(min_distance)

    return tuple(correlation_distances_list)
