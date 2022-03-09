import numpy
from numba import jit

from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class VarianceStabilisationTransform(ImageTransformBase):
    """Variance Stabilization Transform (VST)
    and other range compression functions.

    Applies either the Yeo-Johnson, Box-Cox, Anscomb or some other VST
    transform to the image. Variance stabilization turns an image with
    Poisson or in general non-Gaussian noise into an image with approximately
    Gaussian noise, which often facilitates denoising.
    Also, in general for images with very skewed histograms with for example
    dark regions of large extent and only a few very bright regions,
    it is often beneficial to apply a sqrt or log transformation before
    denoising.
    """

    preprocess_description = (
        "Apply transform" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Undo transform" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        mode: str = 'anscomb',
        leave_as_float: bool = True,
        priority: float = 0.11,
        **kwargs,
    ):

        """
        Constructs a Variance Stabilisation Transform

        Parameters
        ----------
        mode : str
            Variance stabilisation mode: 'yeo-johnson', 'box-cox',
            'quantile', 'anscomb', 'log', 'sqrt' and 'identity'.
            Our tests show that Anscomb seems to work best for
            typical microscopy images. However, your mileage may
            vary. Images with strong contrast and very little
            noise might not need variance stabilisation, but could
            still benefit from value compression (log or sqrt).
            The advantage of the Yeo-Johnson and Box-Cox transforms
            is that they are adaptive, the transform is adjusted
            to match the input distribution.

        leave_as_float : bool
            Does not attempt to cast back to original non-flloat data
            type (8, 16, 32 bit integer), but instead leaves as 32 bit float.

        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)

        self.mode = mode
        self.leave_as_float = leave_as_float

        self._original_dtype = None
        self._min = None
        self._max = None
        self._transform: TransformerMixin = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_original_dtype']
        del state['_min']
        del state['_max']
        del state['_transform']
        return state

    def __str__(self):
        return f'{type(self).__name__}' f' (mode={self.mode})'

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Stabilising variance ({self.mode}) for array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            # Let's ensure we are working with floats:
            self._original_dtype = array.dtype
            array = array.astype(numpy.float32, copy=False)

            if self.mode == 'identity':
                new_array = array.copy()
            elif self.mode == 'sqrt':
                min_value = array.min()
                new_array = _f_sqrt(array, min_value)
                self._min_value = min_value
            elif self.mode == 'log':
                min_value = array.min()
                new_array = _f_log(array, min_value)
                self._min_value = min_value
            elif self.mode == 'anscomb':
                min_value = array.min()
                new_array = _f_anscomb(array, min_value)
                self._min_value = min_value
            elif self.mode == 'yeo-johnson' or self.mode == 'box-cox':

                # If the image has negative values, or too small variance, we fallback to yeo-johnson:
                mode = (
                    'yeo-johnson'
                    if array.min() < 0 or numpy.var(numpy.log1p(array.ravel())) < 1e-6
                    else self.mode
                )
                lprint(f"Actual mode used: {mode} ")

                # We prepare array to make it compatible to sklearn API and save shape:
                shape = array.shape
                array = array.ravel()[:, numpy.newaxis]

                try:
                    # Instantiate sklearn power transform:
                    power_transform = PowerTransformer(method=mode, standardize=True)

                    numpy.seterr(all='raise')
                    # Fit tranform:
                    power_transform.fit(array)
                except (Warning, FloatingPointError):
                    lprint(
                        f"VST {mode} failed for some numerical reasons, falling back to yeo-johnson."
                    )
                    mode = 'yeo-johnson'

                    # Instantiate sklearn power transform:
                    power_transform = PowerTransformer(method=mode, standardize=True)

                    # Fit tranform:
                    power_transform.fit(array)

                # Apply transform:
                new_array = power_transform.transform(array)

                # Reshape array:
                new_array = numpy.reshape(new_array, newshape=shape)

                # Save transform:
                self._transform = power_transform
            elif self.mode == 'quantile':

                # We prepare array to make it compatible to sklearn API and save shape:
                shape = array.shape
                array = array.ravel()[:, numpy.newaxis]

                # Instantiate sklearn quantile transform:
                power_transform = QuantileTransformer(
                    n_quantiles=1024, output_distribution='normal'
                )

                # Fit tranform:
                power_transform.fit(array)

                # Apply transform:
                new_array = power_transform.transform(array)

                # Reshape array:
                new_array = numpy.reshape(new_array, newshape=shape)

                # Save transform:
                self._transform = power_transform
            else:
                raise ValueError(f"Unsupported or incompatible VST mode: {self.mode}")

            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Applying inverse variance stabilisation transform ({self.mode}) for array of shape: {array.shape} and dtype: {array.dtype}"
        ):

            if self.mode == 'identity':
                new_array = array.copy()
            elif self.mode == 'sqrt':
                new_array = _i_sqrt(array, self._min_value)
            elif self.mode == 'log':
                new_array = _i_log(array, self._min_value)
            elif self.mode == 'anscomb':
                new_array = _i_anscomb(array, self._min_value)
            elif (
                self.mode == 'yeo-johnson'
                or self.mode == 'box-cox'
                or self.mode == 'quantile'
            ):

                # We prepare array to make it compatible to sklearn API and save shape:
                shape = array.shape
                array = array.ravel()[:, numpy.newaxis]

                # Apply inverse transform:
                new_array = self._transform.inverse_transform(array)

                # Reshape array:
                new_array = numpy.reshape(new_array, newshape=shape)

                # reset transform:
                self._transform = None
            else:
                raise ValueError(f"Unsupported VST mode: {self.mode}")

            if not self.leave_as_float:
                # convert back to original dtype:
                new_array = new_array.astype(dtype=self._original_dtype, copy=False)

            return new_array


@jit(nopython=True, parallel=True, error_model='numpy')
def _f_sqrt(image, min_value):
    return numpy.sqrt(image - min_value)


@jit(nopython=True, parallel=True, error_model='numpy')
def _i_sqrt(image, min_value):
    return image ** _F(2) + min_value


@jit(nopython=True, parallel=True, error_model='numpy')
def _f_log(image, min_value):
    return numpy.log1p(image - min_value)


@jit(nopython=True, parallel=True, error_model='numpy')
def _i_log(image, min_value):
    return numpy.exp(image) - _F(1) + min_value


@jit(nopython=True, parallel=True, error_model='numpy')
def _f_anscomb(image, min_value):
    return _F(2) * numpy.sqrt(image - min_value + _F(0.375))


@jit(nopython=True, parallel=True, error_model='numpy')
def _i_anscomb(image, min_value):
    return (image / _F(2)) ** _F(2) + min_value - _F(0.375)


@jit(nopython=True, error_model='numpy')
def _F(x):
    return numpy.asarray(x, dtype=numpy.float32)
