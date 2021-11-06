import numpy as np
import tensorflow as tf
from scipy.signal.signaltools import (
    _inputs_swap_needed,
    _init_freq_conv_axes,
    _apply_conv_mode,
)

from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv


class ImageTranslatorLRDeconvTensorflow(ImageTranslatorLRDeconv):
    """Lucy Richardson Deconvolution, TensorFlow backend"""

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
            320 ** 3 if self.max_voxels_per_tile is None else self.max_voxels_per_tile
        )

    def _convert_array_format_in(self, input_image):
        return input_image

    def _convert_array_format_out(self, output_image):
        return output_image

    def _get_convolution_method(self, input_image, psf_kernel):
        """Method to get convolution method.

        Parameters
        ----------
        input_image : array_like
        psf_kernel : array_like

        Returns
        -------
        convolve

        """
        return self._convolution

    def _get_pad_method(self):
        """Method to get pad method.

        Returns
        -------
        pad

        """
        import tensorflow

        return tensorflow.pad

    def _convolution(self, in1, in2, mode="full"):
        volume = np.asarray(in1)
        kernel = np.asarray(in2)

        if volume.ndim == kernel.ndim == 0:
            return volume * kernel
        elif volume.ndim != kernel.ndim:
            raise ValueError("volume and kernel should have the same dimensionality")

        if _inputs_swap_needed(mode, volume.shape, kernel.shape):
            # Convolution is commutative; order doesn't have any effect on output
            volume, kernel = kernel, volume

        print("volume shape: ", volume.shape)
        print("kernel shape: ", kernel.shape)

        out = self.fft_convolve(volume, kernel, mode=mode)

        return out.astype(np.result_type(volume, kernel), copy=False)

    def fft_convolve(self, in1, in2, mode="full", axes=None):
        """FFT convolve

        Parameters
        ----------
        in1
        in2
        mode
        axes

        """

        in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

        s1 = np.asarray(in1.shape)
        s2 = np.asarray(in2.shape)

        shape = tuple(s1 + s2 - 1)

        if not len(axes):
            return in1 * in2

        fft, ifft = self.get_right_fft(in1.ndim), self.get_right_ifft(in1.ndim)

        sp1 = fft(in1, shape)
        sp2 = fft(in2, shape)

        ret = ifft(sp1 * sp2, shape)
        ret = ret.numpy()

        return _apply_conv_mode(ret, s1, s2, mode, axes)

    def get_right_fft(self, ndim):
        if ndim == 1:
            return tf.signal.rfft
        elif ndim == 2:
            return tf.signal.rfft2d
        elif ndim == 3:
            return tf.signal.rfft3d
        else:
            raise Exception("Only convolutions up to 3-dimensions allowed!")

    def get_right_ifft(self, ndim):
        if ndim == 1:
            return tf.signal.irfft
        elif ndim == 2:
            return tf.signal.irfft2d
        elif ndim == 3:
            return tf.signal.irfft3d
        else:
            raise Exception("Only convolutions up to 3-dimensions allowed!")
