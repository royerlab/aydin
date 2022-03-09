from typing import Union, Sequence

import numpy
from numba import jit

from numpy.typing import ArrayLike

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class AttenuationTransform(ImageTransformBase):
    """Axis-aligned attenuation correction

    Corrects intensity attenuation of an image along a given list of axis.
    This is usefull to correct for signal attenuation over time or along
    space. Currently only linear attenuation is supported. More modes on the
    way.
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
        """
        Constructs a Attenuation Corrector

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
            are sorted and applied in ascending order during preprocesing and in
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

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_original_dtype']
        del state['_corrections']
        del state['_axis']
        return state

    def __str__(self):
        return f'{type(self).__name__} (mode={self.mode})'

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Correcting attenuation for array of shape: {array.shape} and dtype: {array.dtype}:"
        ):

            self._axis = self.axis
            if self._axis is None:
                self._axis = list(range(0, array.ndim))

            if type(self._axis) == int:
                self._axis = [self._axis]

            self._original_dtype = array.dtype

            # Allocate new array to store result:
            new_array = array.astype(dtype=numpy.float32, copy=True)

            # Reset cache for undoing transforms:
            self._corrections = {}

            # Iterating over dimensions to correct:
            for axis in self._axis:
                lprint(f"Correcting along axis: {axis}")
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

        if not self.do_postprocess:
            return array

        with lsection(
            f"Reapplying attenuation for array of shape: {array.shape} and dtype: {array.dtype}:"
        ):

            # Allocate new array to store result:
            new_array = array.astype(dtype=numpy.float32, copy=True)

            for axis in reversed(self._axis):
                lprint(f"Correcting along axis: {axis}")
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
    image *= factor
