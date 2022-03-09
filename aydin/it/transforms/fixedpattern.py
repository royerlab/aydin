import itertools
from typing import List, Tuple, Any, Sequence, Union
import numpy

from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class FixedPatternTransform(ImageTransformBase):
    """Fixed Axis-Aligned Pattern Suppression

    Suppresses fixed, axis aligned, offset patterns along any combination of
    axis. Given a list of lists of axis that defines axis-aligned volumes,
    intensity fluctuations of these volumes are stabilised. You can suppress
    intensity fluctuation over time, suppress fixed offsets per pixel over
    time, suppress intensity fluctuations per row, per column, and more...

    For example, assume an image with dimensions tyx (t+2D), and you want to
    suppress fluctuations of intensity along the t axis, then you provide:
    axes=[[0]] (or simply 0 or [0]) which means that the average intensity
    for all planes along t (axis=0) will be stabilised. If instead you want
    to suppress some fixed background offset over xy planes, then you do:
    axes=[[1,2]]. If you want to do both, then you use: axes=[[0], [1,
    2]]. Please note that these corrections are applied in the order
    specified by the list of axis combinations. It is not recommended to
    reapply the pattern after denoising, unless the pattern itself is of
    value and is not considered noise.
    """

    preprocess_description = (
        "Axis-aligned pattern suppression" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Reapplies pattern" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = False

    def __init__(
        self,
        axes: Union[None, int, Sequence[int], List[Sequence[int]]] = None,
        percentile: float = 1,
        sigma: float = 0.5,
        priority: float = 0.09,
        **kwargs,
    ):

        """
        Constructs a Background Correction

        Parameters
        ----------
        axes : Union[None, int, Sequence[int], List[Sequence[int]]]
            List of axis combinations. The order provided
            is the order in which the corrections are applied.
            If None the axes are automatically determined.
        percentile : float
            Percentile value used for estimating brightness.
        sigma : float
            Sigma of the Gaussian filter applied on the
            detected pattern. The higher this value the
            less high-frequency fluctuations are corrected.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)

        # In case a single integer is passed:
        if axes is not None and type(axes) == int:
            axes = list((axes,))

        # In case only one axis combination is given:
        if axes is not None and len(axes) > 0 and type(axes[0]) == int:
            axes = list((axes,))

        # normalise to tuple:
        axes = (
            [tuple(ac) for ac in axes] if axes is not None and len(axes) > 0 else axes
        )

        self.axis_combinations = axes
        self.percentile = percentile
        self.sigma = sigma
        self._corrections = {}

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_corrections']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (percentile={self.percentile},'
            f' sigma={self.sigma})'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Removing axis-aligned fixed patterns for array of shape: {array.shape} and dtype: {array.dtype}:"
        ):
            self._original_dtype = array.dtype
            new_array = array.astype(dtype=numpy.float32, copy=True)

            overall_value = numpy.percentile(
                new_array, q=self.percentile, keepdims=True
            )

            if self.axis_combinations is None:
                # Default:
                axis_combinations = _all_axis_combinations(array.ndim)
            else:
                # Invert the meaning of the axis:
                axis_combinations = (
                    tuple((a for a in range(array.ndim) if a not in ac))
                    for ac in self.axis_combinations
                )

            self._axis_combinations = axis_combinations
            self._corrections = {}

            for axis_combination in axis_combinations:
                lprint(f"Suppressing variations across hyperplane: {axis_combination}")
                value = numpy.percentile(
                    new_array, q=self.percentile, axis=axis_combination, keepdims=True
                )
                value = gaussian_filter(value, sigma=self.sigma)

                correction = overall_value - value
                new_array += correction

                self._corrections[axis_combination] = correction

            self.overall_value = overall_value

            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Adding back axis-aligned fixed pattern to array of shape: {array.shape} and dtype: {array.dtype}:"
        ):

            # Allocate new array to store result:
            new_array = array.astype(dtype=numpy.float32, copy=True)

            for axis_combination in reversed(self._axis_combinations):
                correction = self._corrections[axis_combination]
                new_array -= correction
                self._corrections[axis_combination] = correction

            new_array = new_array.astype(self._original_dtype, copy=False)

            return new_array


def _axis_combinations(ndim: int, n: int) -> List[Tuple[Any, ...]]:
    return list(itertools.combinations(range(ndim), n))


def _all_axis_combinations(ndim: int):
    axis_combinations = []
    for dim in range(1, ndim):
        combinations = _axis_combinations(ndim, dim)
        axis_combinations.extend(combinations)
    return axis_combinations
