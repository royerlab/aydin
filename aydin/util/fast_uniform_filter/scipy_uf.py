import numpy
from scipy.ndimage import uniform_filter


def scipy_uniform_filter(
    image, size=3, output=None, mode="nearest", cval=0.0, origin=0
):
    # Save original image dtype:
    original_dtype = image.dtype

    # Scipy does not support float16 yet:
    dtype = numpy.float32 if original_dtype == numpy.float16 else original_dtype
    image = image.astype(dtype=dtype, copy=False)

    output = uniform_filter(image, size=size, mode=mode, cval=cval, origin=origin)

    output = output.astype(dtype=original_dtype, copy=False)

    return output
