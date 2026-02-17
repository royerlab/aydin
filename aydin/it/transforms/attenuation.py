"""Axis-aligned attenuation correction transform.

Corrects intensity attenuation along specified axes using linear trend
fitting (Theil-Sen regression). Useful for compensating signal decay over
time or depth in microscopy images.
"""

from typing import Sequence, Union

import numpy
from numba import jit
from numpy.typing import ArrayLike

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import aprint, asection


class AttenuationTransform(ImageTransformBase):
    """Axis-aligned attenuation correction.

    Corrects intensity attenuation of an image along a given list of axes.
    This is useful to correct for signal attenuation over time (e.g.
    <a href='https://en.wikipedia.org/wiki/Photobleaching'>photobleaching</a>)
    or along space (e.g. depth-dependent signal loss in
    <a href='https://en.wikipedia.org/wiki/Light_sheet_fluorescence_microscopy'>light-sheet</a>
    or <a href='https://en.wikipedia.org/wiki/Confocal_microscopy'>confocal</a>
    microscopy). The correction uses robust
    <a href='https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator'>Theil-Sen regression</a>
    to estimate a linear trend along each specified axis and divides
    it out. It is generally not recommended to reapply the attenuation
    after denoising, unless the attenuation profile itself is meaningful.
    <notgui>
    """

    preprocess_description = (
        "Suppresses axis-aligned attenuation"
        + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Reapplies attenuation" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = False

    def __init__(
        self,
        axes: Union[None, int, Sequence[int]] = None,
        mode: str = 'linear',
        priority: float = 0.321,
        **kwargs,
    ):
        """Construct an AttenuationTransform.

        Parameters
        ----------
        axes : Union[None, int, Sequence[int]]
            Axis or list of axes over which to correct attenuation.
            If None the axes are automatically determined.

        mode : str
            Attenuation fitting mode, only currently supported: 'linear'

        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocessing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.axis = (
            ([tuple(ac) for ac in axes] if type(axes) is not int else (axes,))
            if axes is not None
            else None
        )
        self.mode = mode
        self._original_dtype = None
        self._corrections = {}
        self._axis = None

        aprint(f"Instantiating: {self}")

    def __getstate__(self):
        """Return picklable state, excluding transient fields.

        Returns
        -------
        dict
            Object state without ``_original_dtype``, ``_corrections``,
            and ``_axis``.
        """
        state = self.__dict__.copy()
        del state['_original_dtype']
        del state['_corrections']
        del state['_axis']
        return state

    def __str__(self):
        """Return a human-readable string representation.

        Returns
        -------
        str
            String showing the class name and mode.
        """
        return f'{type(self).__name__} (mode={self.mode})'

    def __repr__(self):
        """Return a detailed string representation.

        Returns
        -------
        str
            Same as ``__str__``.
        """
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        """Correct intensity attenuation along the specified axes.

        Parameters
        ----------
        array : ArrayLike
            Input image array.

        Returns
        -------
        numpy.ndarray
            Attenuation-corrected image as float32.
        """

        with asection(
            f"Correcting attenuation for array of shape: "
            f"{array.shape} and dtype: {array.dtype}:"
        ):

            self._axis = self.axis
            if self._axis is None:
                self._axis = list(range(0, array.ndim))

            if isinstance(self._axis, int):
                self._axis = [self._axis]

            self._original_dtype = array.dtype

            # Allocate new array to store result:
            new_array = array.astype(dtype=numpy.float32, copy=True)

            # Reset cache for undoing transforms:
            self._corrections = {}

            # Iterating over dimensions to correct:
            for axis in self._axis:
                aprint(f"Correcting along axis: {axis}")
                if array.shape[axis] <= 1:
                    continue

                # Which axis do we need to compute the medians over?
                axis_to_compute_medians = tuple(
                    i for i in range(0, array.ndim) if i != axis
                )
                medians_along_axis = numpy.median(
                    array, axis=axis_to_compute_medians
                ).astype(numpy.float32, copy=True)

                # Compute trend:
                trend = self._trend_fit(medians_along_axis)

                # Make the trend array ndimensional:
                trend_nd_slice_tuple = tuple(
                    None if i != axis else slice(None) for i in range(0, array.ndim)
                )
                trend_nd = trend[trend_nd_slice_tuple]

                # Compute correction:
                trend_max = numpy.max(trend)
                correction = trend_max / trend_nd
                correction = correction.astype(dtype=numpy.float32, copy=True)

                # fast in-place multiply:
                _in_place_multiply(new_array, correction)

                # remember correction to be able to undo it:
                self._corrections[axis] = correction

            return new_array

    def _trend_fit(self, medians_along_axis):
        """Fit a trend to the median values along an axis.

        Parameters
        ----------
        medians_along_axis : numpy.ndarray
            1D array of median values along the attenuation axis.

        Returns
        -------
        numpy.ndarray
            Fitted trend values.

        Raises
        ------
        ValueError
            If the attenuation correction mode is not supported.
        """
        if self.mode == 'linear':
            # prepare data for Thiel-Sen fitting:
            X = numpy.arange(0, len(medians_along_axis), 1).reshape(-1, 1)
            y = medians_along_axis
            # Thiel-Sen fitting:
            from sklearn.linear_model import TheilSenRegressor

            reg = TheilSenRegressor(random_state=0).fit(X, y)
            # Compute the estimated trend:
            trend = reg.predict(X)
            return trend
        else:
            raise ValueError(f"Mode '{self.mode}' for attenuation correction unknown!")

    def postprocess(self, array: ArrayLike):
        """Reapply the attenuation that was corrected during preprocessing.

        Parameters
        ----------
        array : ArrayLike
            Denoised image array.

        Returns
        -------
        numpy.ndarray
            Image with original attenuation restored.
        """

        if not self.do_postprocess:
            return array

        with asection(
            f"Reapplying attenuation for array of shape: "
            f"{array.shape} and dtype: {array.dtype}:"
        ):

            # Allocate new array to store result:
            new_array = array.astype(dtype=numpy.float32, copy=True)

            for axis in reversed(self._axis):
                aprint(f"Correcting along axis: {axis}")
                correction = self._corrections[axis]
                inverse_correction = 1.0 / correction
                inverse_correction = inverse_correction.astype(dtype=correction.dtype)
                _in_place_multiply(new_array, inverse_correction)

            new_array = new_array.astype(self._original_dtype, copy=False)

            return new_array


@jit(
    nopython=True,
    parallel=False,
    error_model='numpy',
    fastmath={'contract', 'afn', 'reassoc'},
)
def _in_place_multiply(image, factor):
    """Multiply an image array in-place by a factor (Numba-accelerated).

    Parameters
    ----------
    image : numpy.ndarray
        Image array to modify in-place.
    factor : numpy.ndarray
        Multiplicative correction factor.
    """
    image *= factor
