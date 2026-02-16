"""Histogram equalisation transform.

Applies histogram equalisation or CLAHE (Contrast Limited Adaptive Histogram
Equalisation) to images with extremely unbalanced histograms, improving
denoising performance. The transform can be reversed during post-processing
(for the 'equalize' mode).
"""

import numpy
from numpy.typing import ArrayLike
from skimage.exposure import cumulative_distribution, equalize_adapthist

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import aprint, asection


class HistogramEqualisationTransform(ImageTransformBase):
    """Histogram equalisation transform.

    For some images with extremely unbalanced histograms, applying histogram
    equalisation will improve results. Two modes are supported: 'equalize'
    (standard histogram equalisation) and 'clahe' (Contrast Limited Adaptive
    Histogram Equalisation).
    <notgui>
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
        """Construct a HistogramEqualisationTransform.

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
            are sorted and applied in ascending order during preprocessing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.mode = mode
        self.scale = scale
        self._cdf = None
        self._bin_centers = None
        self._original_dtype = None

        aprint(f"Instantiating: {self}")

    def __getstate__(self):
        """Return picklable state, excluding transient fields.

        Returns
        -------
        dict
            Object state without ``_cdf``, ``_bin_centers``, and
            ``_original_dtype``.
        """
        state = self.__dict__.copy()
        del state['_cdf']
        del state['_bin_centers']
        del state['_original_dtype']
        return state

    def __str__(self):
        """Return a human-readable string representation.

        Returns
        -------
        str
            String showing the class name, mode, and scale.
        """
        return f'{type(self).__name__}' f' (mode={self.mode},' f' scale={self.scale})'

    def __repr__(self):
        """Return a detailed string representation.

        Returns
        -------
        str
            Same as ``__str__``.
        """
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        """Apply histogram equalisation to the image.

        Parameters
        ----------
        array : ArrayLike
            Input image array.

        Returns
        -------
        numpy.ndarray
            Histogram-equalised image.
        """

        with asection(
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
        """Undo histogram equalisation applied during preprocessing.

        For 'equalize' mode, applies inverse interpolation. For 'clahe'
        mode, the inverse is not currently supported and the array is
        returned unchanged.

        Parameters
        ----------
        array : ArrayLike
            Denoised image array.

        Returns
        -------
        numpy.ndarray
            Image with original histogram characteristics restored.
        """

        if not self.do_postprocess:
            return array

        with asection(
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
    """Interpolate image values using a piecewise linear mapping.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    x : numpy.ndarray
        Source values for interpolation.
    y : numpy.ndarray
        Target values for interpolation.

    Returns
    -------
    numpy.ndarray
        Interpolated image with same shape as input.
    """
    out = numpy.interp(image.flat, x, y)
    return out.reshape(image.shape)
