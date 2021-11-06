import numpy
import scipy
from scipy.fft import next_fast_len
from scipy.signal.signaltools import _init_freq_conv_axes, _apply_conv_mode

from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv


class ImageTranslatorLRDeconvScipy(ImageTranslatorLRDeconv):
    """Lucy Richardson Deconvolution, SciPy backend"""

    def __init__(self, *args, **kwargs):
        """Constructs a Lucy Richardson deconvolution image translator.

        Parameters
        ----------
        psf_kernel : numpy.typing.ArrayLike
            2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        max_num_iterations : int
        clip : bool
        backend : str
            Computation backend selection.
        kwargs : dict
        """
        super().__init__(*args, **kwargs)

        self.max_voxels_per_tile = (
            512 ** 3 if self.max_voxels_per_tile is None else self.max_voxels_per_tile
        )

    def _convert_array_format_in(self, input_image):
        return input_image

    def _convert_array_format_out(self, output_image):
        return output_image

    def _get_convolution_method(self, input_image, psf_kernel):
        """
        Method to get convolution method.

        Parameters
        ----------
        input_image : array_like
        psf_kernel : array_like

        Returns
        -------
        convolve

        """

        return _fftconvolve

    def _get_pad_method(self):
        """
        Method to get pad method.

        Returns
        -------
        pad

        """
        import numpy

        return numpy.pad


def _fftconvolve(in1, in2, mode="full", axes=None):
    in1 = numpy.asarray(in1)
    in2 = numpy.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return numpy.array([])

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    if not len(axes):
        return in1 * in2

    complex_result = in1.dtype.kind == 'c' or in2.dtype.kind == 'c'

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = scipy.fft.rfftn, scipy.fft.irfftn
    else:
        fft, ifft = scipy.fft.fftn, scipy.fft.ifftn

    sp1 = fft(in1, fshape, axes=axes, workers=-1)
    sp2 = fft(in2, fshape, axes=axes, workers=-1)

    ret = ifft(sp1 * sp2, fshape, axes=axes, workers=-1)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret
