import numpy
from scipy.ndimage import gaussian_filter

from aydin.util.denoise_nd.denoise_nd import extend_nd


def demo_denoise_nd():
    # raw function that only supports 2D images:
    def function(image, sigma):
        if image.ndim != 2:
            raise RuntimeError("Function only supports arrays of dimensions 2")
        return gaussian_filter(image, sigma)

    # extended function that supports all dimension (with all caveats associated to how we actually do this extension...)
    @extend_nd(available_dims=[2])
    def extended_function(image, sigma):
        return function(image, sigma)

    image = numpy.zeros((32, 5, 64))
    image[16, 2, 32] = 1

    denoised = extended_function(image, sigma=1)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')


demo_denoise_nd()
