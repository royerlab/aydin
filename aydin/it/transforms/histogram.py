import numpy

from numpy.typing import ArrayLike
from skimage.exposure import equalize_adapthist, cumulative_distribution

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class HistogramEqualisationTransform(ImageTransformBase):
    """Histogram Equalisation

    For some images with extremely unbalanced histograms, applying histogram
    equalisation will improve results. Two modes are supported: 'equalize',
    and 'clahe'.
    """

    preprocess_description = (
        "Apply histogram equalisation" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Undo histogram equalisation" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        mode: str = 'equalize',
        scale: float = 1.0 / 8,
        priority: float = 0.12,
        **kwargs,
    ):

        """
        Constructs a Histogram Transform

        Parameters
        ----------
        mode : str
            Two modes are supported: 'equalize', and 'clahe'.
        scale : float
            Scale of the kernel expressed relatively to the
            size of the image, values are within [0,1].
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.mode = mode
        self.scale = scale
        self._cdf = None
        self._bin_centers = None
        self._original_dtype = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cdf']
        del state['_bin_centers']
        del state['_original_dtype']
        return state

    def __str__(self):
        return f'{type(self).__name__}' f' (mode={self.mode},' f' scale={self.scale})'

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Equalises histogram for array of shape: {array.shape} and dtype: {array.dtype}"
        ):

            self._original_dtype = array.dtype
            array = array.astype(numpy.float32, copy=False)

            if self.mode == 'equalize':
                self._cdf, self._bin_centers = cumulative_distribution(array)
                new_array = _interpolation(array, self._bin_centers, self._cdf)
            elif self.mode == 'clahe':
                kernel_size = tuple(s / self.scale for s in array.shape)
                new_array = equalize_adapthist(array, kernel_size=kernel_size)
            else:
                raise ValueError(
                    f"Unsupported mode for histogram transform: {self.mode}"
                )

            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Undoing histogram equalisation for array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            array = array.astype(numpy.float32, copy=False)

            if self.mode == 'equalize':
                new_array = _interpolation(array, self._cdf, self._bin_centers)
            elif self.mode == 'clahe':
                # Inverse not supported yet:
                new_array = array
            else:
                raise ValueError(
                    f"Unsupported mode for histogram transform: {self.mode}"
                )

            # cast back to original dtype:
            new_array = new_array.astype(self._original_dtype, copy=False)

            return new_array


def _interpolation(image, x, y):
    out = numpy.interp(image.flat, x, y)
    return out.reshape(image.shape)
