from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.util.log.log import lsection
from aydin.util.misc.progress_bar import ProgressBar


class LucyRichardson:
    """LucyRichardson Service.

    Parameters
    ----------
    backend : str
        Opacity of the layer visual, between 0.0 and 1.0.
    psf_kernel : array_like
        2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
    max_num_iterations : int
    backend : str
        Computation backend selection.
    """

    def __init__(
        self, *, psf_kernel=None, max_num_iterations=20, backend='scipy', **kwargs
    ):

        # default:
        deconvolution_class = ImageTranslatorLRDeconvScipy
        if backend == 'scipy':
            deconvolution_class = globals()['ImageTranslatorLRDeconvScipy']
        elif backend == 'cupy':
            deconvolution_class = globals()['ImageTranslatorLRDeconvCupy']
        elif backend == 'gputools':
            deconvolution_class = globals()['ImageTranslatorLRDeconvGputools']

        self.it = deconvolution_class(
            psf_kernel=psf_kernel, max_num_iterations=max_num_iterations, **kwargs
        )

    def train(self, input_image, target_image, **kwargs):
        """Method to prepare ImageTranslatorLRDeconv instance.
        For now there is no actual training going on in lower level calls.
        However, calling this method needed to keep normal it compute flow.

        Parameters
        ----------
        input_image : numpy.ndarray
        target_image : numpy.ndarray

        """
        with lsection("LucyRichardson training is starting..."):
            self.it.train(
                input_image,
                target_image,
                train_valid_ratio=kwargs['train_valid_ratio']
                if 'train_valid_ratio' in kwargs
                else 0.1,
                callback_period=kwargs['callback_period']
                if 'callback_period' in kwargs
                else 3,
                jinv=kwargs['jinv'] if 'jinv' in kwargs else None,
            )

    def deconvolve(self, input_image, **kwargs):
        """Method to deconvolve an image by LucyRichardson technique.

        Parameters
        ----------
        input_image : numpy.ndarray

        Returns
        -------
        response : numpy.ndarray
            deconvolved image.

        """
        with lsection("LucyRichardson translation is starting..."):
            response = self.it.translate(
                input_image,
                tile_size=kwargs['tile_size'] if 'tile_size' in kwargs else None,
            )
            response = response.astype(input_image.dtype, copy=False)

            return response


def lucyrichardson(input_image, *, batch_axes=None, chan_axes=None, backend=None):
    """Method to denoise an image with trained Noise2Self.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image to deconvolve.
    batch_axes : array_like, optional
        Indices of batch axes.
    chan_axes : array_like, optional
        Indices of channel axes.

    Returns
    -------
    response : numpy.ndarray

    """
    pbar = ProgressBar(total=100)
    lr = LucyRichardson(pbar, backend=backend)

    # Train
    lr.train(input_image)

    # Denoise
    deconvolved = lr.deconvolve(input_image)

    return deconvolved
