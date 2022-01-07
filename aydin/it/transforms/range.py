from typing import Optional

import numpy
from numpy.typing import ArrayLike

from aydin.it.normalisers.base import NormaliserBase
from aydin.it.normalisers.minmax import MinMaxNormaliser
from aydin.it.normalisers.percentile import PercentileNormaliser
from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class RangeTransform(ImageTransformBase):
    """Range Normalisation

    Images come in all sorts of formats, and pixels can be represented as 8,
    16, 32 bit integers as well as 16, 32, 64 bit floats. More crucially,
    the actual range of values can vary potentially causing difficulties for
    denoising algorithms. For these reasons and more, it is almost always
    recommended to normalise images to a range within [0, 1] and represented
    as 32 bit floats (sufficient for most cases and ideal for aydin).
    Finally, there are two different normalisation modes. The first 'minmax'
    simply finds the min and max values in the image and uses that to rescale
    to [0, 1]. However, there are sometimes outlier values that are isolated
    and completely out of context which would skew the normalisation.
    Therefore, we also have a 'percentile' mode that uses percentiles to
    determine the range -- typically 1% for min value and 99% for max value.
    Optionally, the image can be left as float after denormalisation,
    and values can be clipped to the range during both normalisation and
    denormalisation.
    """

    preprocess_description = (
        "Range normalisation" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Range denormalisation" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        mode: str = 'minmax',
        percentile: Optional[float] = None,
        force_float_datatype: bool = False,
        clip: bool = True,
        priority: float = 0.2,
        **kwargs,
    ):

        """
        Constructs a Range Transform

        Parameters
        ----------
        mode : str
            Range normalisation mode: 'minmax' or 'percentile'
        percentile : Optional[float]
            Percentile value for the 'percentile' mode.
            If None the percentile value is automatically chosen
            based on the number of voxels in the image.
        force_float_datatype: bool
            After denormalisation the values are left as 32 bit
            floats instead of being converted back to the original
            data type. If False the best setting is automatically chosen.
        clip: bool
            Clips values outside of the range during normalisation
            and denormalisation.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)

        self.mode = mode
        self.percentile = percentile
        self.force_float_datatype = force_float_datatype
        self.clip = clip
        self._normaliser: NormaliserBase
        self._min_value = None
        self._max_value = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_normaliser']
        del state['_min_value']
        del state['_max_value']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (mode={self.mode},'
            f' percentile={self.percentile},'
            f' leave_as_float={self.force_float_datatype},'
            f' clip={self.clip} )'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Normalizing value range ({self.mode}) for array of shape: {array.shape} and dtype: {array.dtype}"
        ):

            self._original_dtype = array.dtype
            array = array.astype(numpy.float32, copy=False)

            if self.mode == 'minmax':
                normaliser = MinMaxNormaliser()
            elif self.mode == 'percentile':
                normaliser = PercentileNormaliser(percentile=self.percentile)

            self._min_value, self._max_value = normaliser.calibrate(array)

            new_array = normaliser.normalise(array)

            self._normaliser = normaliser
            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Denormalizing value range ({self.mode}) for array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            force_float_datatype = self.force_float_datatype

            # Let's figure out if it is reasonable to keep the denoised data as float:
            if force_float_datatype is False and numpy.issubdtype(
                self._original_dtype, numpy.integer
            ):
                range = abs(self._max_value - self._min_value)
                if range < 128:
                    force_float_datatype = True

            new_array = self._normaliser.denormalise(
                array, leave_as_float=force_float_datatype, clip=self.clip
            )
            new_array = new_array.astype(self._original_dtype, copy=False)

            return new_array
