import math
from typing import Union, Optional, Sequence

import numpy
import scipy

from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lprint, lsection


class MotionStabilisationTransform(ImageTransformBase):
    """Motion Stabilisation

    Denoising is more effective if signal-correlated voxels are close to each
    other. When a 2D+t or 3D+t timelapse has shifts between time points,
    pixels that should be close to each other from one time point to the next
    are now further away from each other. Worse, this relative placement
    often varies over time. This complicates denoising and typically leads to
    more blurry denoised images. Thus, stabilizing a timelapse before
    denoising is recommended to improve denoising performance. Currently,
    we assume that all frames can be registered to a common reference frame,
    and thus that all images have a common background that can be used for
    registration. For completeness, multiple axis can be specified and the
    correction is applied along each in sequence.(advanced)
    """

    preprocess_description = (
        "Stabilise motion" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Reapply motion" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = False

    def __init__(
        self,
        axes: Union[None, int, Sequence[int]] = 0,
        sigma: float = 3,
        pad: bool = False,
        crop: bool = False,
        pad_mode: str = 'min_constant',
        max_pixel_shift: Optional[int] = None,
        reference_index: Optional[int] = None,
        priority: float = 0.45,
        **kwargs,
    ):
        """
        Constructs a Motion Corrector

        Parameters
        ----------
        axes : Union[None, int, Sequence[int]]
            Index of time axis. If None the axes
            are determined automatically.
        sigma : float
            Sigma for Gaussian filtering used to
            facilitate registration.
        pad : bool
            Pads image before applying stabilisation,
            default value is True.
        crop : bool
            Crops image after undoing stabilisation,
            default value is True
        pad_mode : str
            Padding mode. Can be: 'mean_constant',
            'min_constant', 'max_constant', 'wrap'.
            Default is 'wrap'.
        max_pixel_shift : Optional[int]
            Maximum correctable motion in pixels.
            If None the maximum shift is automatically
            determined.
        reference_index : Optional[int]
            Index of image used as reference to register
            all others. If None the reference image is
            automatically determined.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.sigma = sigma
        self.axes = (
            None if axes is None else ((axes,) if isinstance(axes, int) else axes)
        )
        self.pad_mode = pad_mode

        self.pad = pad
        self.crop = crop

        self.center = False
        self.max_pixel_shift = max_pixel_shift
        self.reference_index = reference_index

        self._shifts = {}
        self._original_dtype = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_shifts']
        del state['_original_dtype']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (pad_mode={self.pad_mode},'
            f' pad={self.pad},'
            f' crop={self.crop},'
            f' center={self.center},'
            f' max_pixel_shift={self.max_pixel_shift},'
            f' reference_index={self.reference_index})'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        with lsection(f"Motion-correcting array of shape: {array.shape}:"):

            self._original_dtype = array.dtype

            # We need a copy because the shift-transfrom is in-place
            array = array.astype(numpy.float32, copy=True)

            axes = range(array.ndim) if self.axes is None else self.axes

            self._shifts = {}
            for axis in axes:
                lprint(f"Correcting along axis: {axis}")
                array = self._permutate(array, axis=axis)
                shifts, mean_shift = _measure_shifts(
                    array,
                    reference_index=self.reference_index,
                    center=self.center,
                    max_pixel_shift=self.max_pixel_shift,
                    mode='com',
                    sigma=self.sigma,
                )
                lprint(f"Mean shift: {mean_shift}")
                array = _shift_transform(
                    array, -shifts, pad=self.pad, crop=False, pad_mode=self.pad_mode
                )
                self._shifts[axis] = shifts
                array = self._depermutate(array, axis=axis)
            return array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(f"Undoing motion-correction for array of shape: {array.shape}:"):

            # We need a copy because the shift-transfrom is in-place
            array = array.astype(numpy.float32, copy=True)

            axes = range(array.ndim) if self.axes is None else self.axes

            for axis in reversed(axes):
                lprint(f"Correcting along axis: {axis}")
                array = self._permutate(array, axis=axis)
                shifts = self._shifts[axis]
                array = _shift_transform(
                    array, shifts, pad=False, crop=self.crop, pad_mode=''
                )
                array = self._depermutate(array, axis=axis)

            # cast back to original dtype:
            array = array.astype(self._original_dtype, copy=False)

            return array

    def _permutate(self, array: ArrayLike, axis: int):
        permutation = self._get_permutation(array, axis=axis)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _depermutate(self, array: ArrayLike, axis: int):
        permutation = self._get_permutation(array, axis=axis, inverse=True)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _get_permutation(self, array: ArrayLike, axis: int, inverse=False):
        permutation = (axis,) + tuple(a for a in range(array.ndim) if a != axis)
        if inverse:
            permutation = numpy.argsort(permutation)
        return permutation


def _shift_transform(array: ArrayLike, shifts, pad, crop, pad_mode='wrap'):
    """ """

    min_shift = abs(numpy.min(shifts, axis=0))
    max_shift = abs(numpy.max(shifts, axis=0))
    lprint(f"min_shift: {min_shift}, max_shift: {max_shift}")

    if pad:
        padding = tuple((mi, ma) for mi, ma in zip(min_shift, max_shift))
        lprint(f"Padding: {padding}")

        value = 0
        value = array.mean() if pad_mode == 'mean_constant' else value
        value = array.min() if pad_mode == 'min_constant' else value
        value = array.max() if pad_mode == 'max_constant' else value
        kwargs = {'constant_values': value} if 'constant' in pad_mode else {}

        pad_mode = 'constant' if 'constant' in pad_mode else pad_mode
        array = numpy.pad(array, pad_width=((0, 0),) + padding, mode=pad_mode, **kwargs)

    for ti, shift in enumerate(shifts):
        lprint(f"Motion correcting {ti} by {shift}")
        array[ti, ...] = numpy.roll(
            array[ti, ...], shift=shift, axis=tuple(range(0, array.ndim - 1))
        )

    if crop:
        crop_slice = (slice(None),) + tuple(
            slice(ma, s - mi)
            for mi, ma, s in zip(min_shift, max_shift, array.shape[1:])
        )
        array = array[crop_slice]

    return array


def _measure_shifts(
    array: ArrayLike,
    reference_index: Optional[int] = None,
    center: bool = False,
    max_pixel_shift: Optional[int] = None,
    mode: str = 'com',
    sigma: float = 7,
):
    shifts = []
    correlations = []

    reference_index = len(array) // 2 if reference_index is None else reference_index
    max_pixel_shift = (
        max(array.shape[1:]) // 3 if max_pixel_shift is None else max_pixel_shift
    )

    for i in range(0, len(array)):
        image = array[i]
        if reference_index >= 0:
            reference_image = array[reference_index]
        elif reference_index < 0:
            reference_image = array[max(0, i + reference_index)]

        shift, correlation = _find_shift(
            image,
            reference_image,
            max_pixel_shift=max_pixel_shift,
            mode=mode,
            sigma=sigma,
        )
        shifts.append(shift)
        correlations.append(correlation)
        lprint(f"Measured shift of {shift} for image {i}.")

    shifts = numpy.array(shifts)
    if reference_index < 0:
        shifts = numpy.cumsum(shifts, axis=0)

    # correlations = numpy.stack(correlations)
    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(array, name='array')
    #     viewer.add_image(correlations, name='correlations')

    # Center shifts:
    if center:
        mean_shift = numpy.round(numpy.mean(shifts, axis=0))
        mean_shift = mean_shift.astype(numpy.int)
        shifts = shifts - mean_shift
    else:
        mean_shift = numpy.array((0,) * (array.ndim - 1))

    # Convert to int:
    shifts = numpy.round(shifts).astype(numpy.int)

    return shifts, mean_shift


def _find_shift(a, b, max_pixel_shift: int = 64, mode: str = 'com', sigma: float = 7):
    # Basic idea: We just need to low-pass filter the heck of it, and it works.
    lprint(f"max_pixel_shift: {max_pixel_shift}, mode: {mode}, sigma: {sigma}")

    # First we blur the input images:
    a = _fast_denoise(a, sigma=sigma)
    b = _fast_denoise(b, sigma=sigma)

    # We compute the phas correlation:
    raw_correlation = _phase_correlation(a, b)

    # We denoise the correlogram itself again:
    correlation = _fast_denoise(raw_correlation, sigma=sigma)
    # correlation = raw_correlation

    # We estimate the noise floor of the correlation:
    empty_region = correlation.copy()
    empty_region_slice = tuple(
        slice(min(s // 2 - 1, max_pixel_shift), -min(s // 2 - 1, max_pixel_shift))
        for s in correlation.shape
    )
    empty_region = empty_region[empty_region_slice]
    noise_floor_level = numpy.percentile(empty_region, q=99.9)

    # we use that floor to clip anything below:
    correlation = correlation.clip(noise_floor_level, math.inf) - noise_floor_level

    # We roll the array and crop it to restrict ourself to the search region:
    correlation = numpy.roll(
        correlation, shift=max_pixel_shift, axis=tuple(range(a.ndim))
    )
    correlation = correlation[(slice(0, 2 * max_pixel_shift),) * a.ndim]

    if mode == 'gfit':

        # This looks fancy and shit but is just bad.

        def gaussian(x, sx=0, sy=0, b=1, a=1):
            return a * numpy.exp(-b * ((x[0] - sx) ** 2 + (x[1] - sy) ** 2))

        x = numpy.arange(0, 2 * max_pixel_shift, 1)
        y = numpy.arange(0, 2 * max_pixel_shift, 1)
        xx, yy = numpy.meshgrid(x, y)
        xdata = numpy.stack([xx, yy]).reshape(2, -1)
        ydata = correlation[xx, yy].reshape(-1)

        popt, pcov = curve_fit(
            gaussian,
            xdata,
            ydata,
            method='trf',
            p0=(0, 0, 1, 1),
            bounds=(
                [-max_pixel_shift, -max_pixel_shift, 0, 0],
                [+max_pixel_shift, +max_pixel_shift, numpy.inf, numpy.inf],
            ),
        )

        shift = popt[:2]

    elif mode == 'com':

        # This is simple, and works brilliantly, even with tons of noise.

        # We use the max as quickly computed proxy for the real center:
        rough_shift = numpy.unravel_index(
            numpy.argmax(correlation, axis=None), correlation.shape
        )

        # We crop further to facilitate center-of-mass estimation:
        fine_window_radius = 4 * sigma
        cropped_correlation_slice = tuple(
            slice(
                max(0, int(rs - fine_window_radius)),
                min(s - 1, int(rs + fine_window_radius)),
            )
            for rs, s in zip(rough_shift, correlation.shape)
        )
        lprint(f"Cropped correlation: {cropped_correlation_slice}")
        cropped_correlation = correlation[cropped_correlation_slice]

        # We compute the signed rough shift
        signed_rough_shift = numpy.array(rough_shift) - max_pixel_shift

        if numpy.all(cropped_correlation == 0):
            # No mass:
            signed_com_shift = numpy.zeros_like(signed_rough_shift)
        else:
            # We compute the center of mass:
            # We take the square to squash small values far from the maximum that are likely noisy...
            signed_com_shift = (
                numpy.array(scipy.ndimage.center_of_mass(cropped_correlation ** 2))
                - fine_window_radius
            )

        # The final shift is the sum of the rough sight plus the fine center of mass shift:
        shift = signed_rough_shift + signed_com_shift

    #
    shift = numpy.nan_to_num(shift)

    return shift, correlation


def _fast_denoise(array: ArrayLike, sigma):
    denoised = gaussian_filter(array, sigma=sigma, mode='wrap')
    return denoised


def _phase_correlation(image, reference_image):
    G_a = scipy.fft.fftn(image, workers=-1)
    G_b = scipy.fft.fftn(reference_image, workers=-1)
    conj_b = numpy.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= numpy.absolute(R)
    r = scipy.fft.ifftn(R, workers=-1).real
    return r
