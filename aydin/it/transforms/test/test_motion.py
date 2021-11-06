import numpy
from skimage.util import random_noise

from aydin.io.datasets import camera, normalise, pollen
from aydin.it.transforms.motion import (
    MotionStabilisationTransform,
    _find_shift,
    _measure_shifts,
    _shift_transform,
)


def test_phase_correlation():
    a = camera()
    b = numpy.roll(camera(), shift=(30, 50), axis=tuple(range(a.ndim)))

    # phase_correlation = _phase_correlation(image=b, reference_image=a)
    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(a, name='a')
    #     viewer.add_image(b, name='b')
    #     viewer.add_image(phase_correlation, name='phase_correlation')

    shift, _ = _find_shift(b, a)

    # print(shift)
    assert (numpy.abs(shift - [30, 50]) < 0.1).all()


def test_measure_shifts_and_transform():
    shifts = tuple((5 * i, int(0.5 * i * i)) for i in range(10))

    # print('')
    # pprint(shifts)

    image = normalise(camera())
    array = numpy.stack(
        [numpy.roll(image, shift=shift, axis=(0, 1)) for shift in shifts]
    )

    measured_shifts, _ = _measure_shifts(array, reference_index=0)

    # pprint(measured_shifts)

    for s, ms in zip(shifts, measured_shifts):
        assert (numpy.array(s) == numpy.array(ms)).all()

    measured_shifts, _ = _measure_shifts(array, reference_index=0, center=False)
    # def _shift_transform(array, shifts, pad, crop, pad_mode='wrap', inverse=False):
    motion_corrected_array = _shift_transform(
        array.copy(), -measured_shifts, pad=False, crop=False
    )

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(array, name='array')
    #     viewer.add_image(motion_corrected_array, name='motion_corrected_array')

    assert (motion_corrected_array == numpy.stack([image for _ in shifts])).all()


def test_correct_uncorrect():
    shifts = tuple((5 * i, int(0.5 * i * i)) for i in range(10))

    # print('')
    # pprint(shifts)

    image = normalise(pollen())[0:256, 0:256]
    array = numpy.stack(
        [add_noise(numpy.roll(image, shift=shift, axis=(0, 1))) for shift in shifts]
    )

    mc = MotionStabilisationTransform(axes=0)

    corrected_array = mc.preprocess(array.copy())
    uncorrected_array = mc.postprocess(corrected_array.copy())

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(array, name='array')
    #     viewer.add_image(corrected_array, name='corrected_array')
    #     viewer.add_image(uncorrected_array, name='uncorrected_array')

    assert array.dtype == uncorrected_array.dtype
    assert (array == uncorrected_array).all()


def add_noise(image, intensity=4, variance=0.4):
    noisy = image
    if intensity is not None:
        noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance)
    noisy = noisy.astype(numpy.float32)
    return noisy
