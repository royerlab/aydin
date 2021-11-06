import cupy
from cupyx.scipy.ndimage import convolve

from aydin.it.deconvolution.lr_deconv import ImageTranslatorLRDeconv


class ImageTranslatorLRDeconvCupy(ImageTranslatorLRDeconv):
    """Richardson Deconvolution, Cupy backend"""

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
        return cupy.asarray(input_image)

    def _convert_array_format_out(self, output_image):
        return cupy.asnumpy(output_image)

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

        # cupy backend does not need extra padding:
        self.padding = False

        def cupy_convolve(in1, in2, mode=None, method=None):
            return convolve(in1, in2, mode='reflect')

        if psf_kernel.size > 500 and not self.no_fft:
            return self._cupy_convolve_fft
        else:
            return cupy_convolve

    def _get_pad_method(self):
        """Method to get pad method.

        Returns
        -------
        pad

        """
        import cupy

        return cupy.pad

    def _cupy_convolve_fft(self, image1, image2, mode=None):
        """Method  for cupy convolve fft.

        Parameters
        ----------
        image1 : array_like
        image2 : array_like

        Returns
        -------
        ret

        """

        import cupy
        import numpy

        # TODO: review if this is needed
        cupy.cuda.set_allocator(None)

        self._debug_allocation("before FFT")

        is_planning_on = cupy.fft.config.enable_nd_planning
        cupy.fft.config.enable_nd_planning = False

        if image1.ndim == image2.ndim == 0:  # scalar inputs
            return image1 * image2
        elif not image1.ndim == image2.ndim:
            raise ValueError("Dimensions do not match.")
        elif image1.size == 0 or image2.size == 0:  # empty arrays
            return cupy.array([])

        s1 = numpy.asarray(image1.shape)
        s2 = numpy.asarray(image2.shape)

        shape = tuple(s1 + s2 - 1)

        fsize = shape  # tuple(int(2 ** math.ceil(math.log2(x))) for x in tuple(shape))

        image1_fft = cupy.fft.rfftn(image1, fsize)
        image2_fft = cupy.fft.rfftn(image2, fsize)
        ret = cupy.fft.irfftn(image1_fft * image2_fft)
        # ret = ret.astype(cupy.float32) #cupy.real(ret)

        fslice = tuple([slice(0, int(sz)) for sz in shape])
        ret = ret[fslice]

        # if mode=='same':
        newshape = cupy.asarray(image1.shape)
        currshape = cupy.array(ret.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

        ret = ret[tuple(myslice)]

        cupy.fft.config.enable_nd_planning = is_planning_on

        del image1_fft
        del image2_fft

        cupy.get_default_memory_pool().free_all_blocks()

        self._debug_allocation("after fft")

        return ret
