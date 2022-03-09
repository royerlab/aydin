import numbers
import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, gaussian_filter

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class HighpassTransform(ImageTransformBase):
    """Highpass Image Simplification

    For images with little noise, applying a high-pass filter can help denoise the image by removing some of the
    image complexity. The low-frequency parts of the image do not need to be denoised because sometimes the challenge
    is disentangling the (high-frequency) noise from the high-frequencies in the image. The scale parameter must be
    chosen with care. The lesser the noise, the smaller the value. Values around 1 work well but must be tuned
    depending on the image. If the scale parameter is too low, some noise might be left untouched. The best is to
    keep the parameter as low as possible while still achieving good denoising performance. It is also possible to
    apply median filtering when computing the low-pass image which helps reducing the impact of outlier voxel values,
    for example salt&pepper noise. Note: when median filtering is on, larger values of sigma (e.g. >= 1) are
    recommended, unless when the level of noise is very low in which case a sigma of 0 (no Gaussian blur) may be
    advantageous. To recover the original denoised image the filtering is undone during post-processing. Note: this
    is ideal for treating <a href='https://en.wikipedia.org/wiki/Colors_of_noise'>'blue' noise</a> that is
    characterised by a high-frequency support.
    """

    preprocess_description = (
        "Remove low spatial frequencies" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Add back low spatial frequencies" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        sigma: float = 1,
        median_filtering: bool = True,
        priority: float = 0.1,
        **kwargs,
    ):
        """
        Constructs a Highpass Transform

        Parameters
        ----------
        sigma : float
            Sigma value of the Gaussian filter used for high-pass filtering.
        median_filtering : bool
            Adds robustness of the filtering in the presence of outliers.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.sigma = sigma
        self.median_filtering = median_filtering
        self._low_pass_image = None
        self._original_dtype = None
        self._min = None
        self._max = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_low_pass_image']
        del state['_original_dtype']
        del state['_min']
        del state['_max']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (sigma={self.sigma}, median_filtering={self.median_filtering})'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Applies high-pass filter of sigma {self.sigma} {'and median filtering' if self.median_filtering else ''} to array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            # Remember min and max:
            self._min = array.min()
            self._max = array.max()

            # remember original dtype:
            self._original_dtype = array.dtype

            # Cast to float if needed:
            array = array.astype(numpy.float32, copy=False)

            # Low-pass filtering:
            self._low_pass_image = self._low_pass_filtering(array)

            # High-pass filtering:
            new_array = array - self._low_pass_image

            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Adds back low-pass frequencies to array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            array = array.astype(numpy.float32, copy=False)

            # Bring back the low frequencies:
            new_array = array + self._low_pass_image

            # If integer type we ensure to stay within original bounds:
            if issubclass(self._original_dtype.type, numbers.Integral):
                new_array = numpy.clip(new_array, self._min, self._max)

            # cast back to original dtype:
            new_array = new_array.astype(self._original_dtype, copy=False)

            # Free memory:
            self._low_pass_image = None

            return new_array

    def _low_pass_filtering(self, array: ArrayLike):

        lprint(f"Sigma for high-pass filter is: {self.sigma}")

        # Median filtering if selected:
        if self.median_filtering:
            array = median_filter(array, size=3)

        # Compute mean:
        mean = numpy.mean(array).astype(numpy.float32)

        # Low-pass filtering:
        if self.sigma > 0:
            array = gaussian_filter(array, sigma=self.sigma) - mean

        return array


def _interpolation(image, x, y):
    out = numpy.interp(image.flat, x, y)
    return out.reshape(image.shape)
