import numpy

from aydin.io.datasets import normalise, pollen, add_noise
from aydin.it.transforms.deskew import DeskewTransform


def demo_deskew():
    shifts = tuple((3 * i, 0) for i in range(100))

    image = normalise(pollen())[0:256, 0:256]
    array = numpy.stack(
        [add_noise(numpy.roll(image, shift=shift, axis=(0, 1))) for shift in shifts]
    )

    sd = DeskewTransform(delta=-3, z_axis=0, skew_axis=1)

    ds_array = sd.preprocess(array)
    s_array = sd.postprocess(ds_array)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(array, name='array')
        viewer.add_image(ds_array, name='ds_array')
        viewer.add_image(s_array, name='s_array')


demo_deskew()
