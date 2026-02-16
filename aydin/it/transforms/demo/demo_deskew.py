"""Demo of the deskew transform for correcting sheared image stacks.

Demonstrates the ``DeskewTransform`` by creating a synthetic sheared
3D stack from a 2D pollen image and applying deskew preprocessing
followed by postprocessing to verify roundtrip fidelity.
"""

import numpy

from aydin.io.datasets import add_noise, normalise, pollen
from aydin.it.transforms.deskew import DeskewTransform


def demo_deskew():
    """Run the deskew transform demo with synthetic sheared data."""
    shifts = tuple((3 * i, 0) for i in range(100))

    image = normalise(pollen())[0:256, 0:256]
    array = numpy.stack(
        [add_noise(numpy.roll(image, shift=shift, axis=(0, 1))) for shift in shifts]
    )

    sd = DeskewTransform(delta=-3, z_axis=0, skew_axis=1)

    ds_array = sd.preprocess(array)
    s_array = sd.postprocess(ds_array)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(array, name='array')
    viewer.add_image(ds_array, name='ds_array')
    viewer.add_image(s_array, name='s_array')
    napari.run()


if __name__ == "__main__":
    demo_deskew()
