import numpy


def check_nd(denoiser):
    shape = (17, 7, 13, 9)

    for ndim in range(1, 5):

        image = numpy.zeros(shape[:ndim])

        try:
            denoiser(image)
            assert True
        except Exception:
            assert False
