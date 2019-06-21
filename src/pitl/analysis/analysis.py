import random
import numpy


def correlation(image, nb_samples=256, max_length=128):


    shape = image.shape
    nb_dim = len(shape)

    corr_list = []

    for dim in range(nb_dim):

        max_length_dim = min(shape[dim],max_length)

        corr_avg = None
        counter = 0

        if shape[dim]==0:
            corr_list = numpy.asarray([0])
            break

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

            if corr_avg is None:
                corr_avg = corr
            else:
                corr_avg += corr


            counter+=1

        corr_avg = corr_avg / counter
        #corr_avg = corr_avg / numpy.sum(corr_avg)
        corr_avg = corr_avg / corr_avg[0]

        corr_list.append(corr_avg)


    return tuple(corr_list)



def correlation_distance(image, percentile = 0.1, nb_samples=100):
    correlation_curves_list = correlation(image, nb_samples=100)

    correlation_distances_list = []

    for correlation_curve in correlation_curves_list:

        length = correlation_curve.shape[0]

        cum_sum = 0
        for distance in range(length):
            cum_sum += correlation_curve[distance]
            if cum_sum>percentile:
                correlation_distances_list.append(distance)
                break

    return tuple(correlation_distances_list)






















