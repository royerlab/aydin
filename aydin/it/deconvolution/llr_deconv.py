from aydin.features.standard_features import StandardFeatureGenerator
from aydin.it.deconvolution.base_deconv import ImageTranslatorDeconvBase
from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


class ImageTranslatorLearnedLRDeconv(ImageTranslatorDeconvBase):
    """Learned Lucy Richardson Deconvolution

    Idea: Instead of just applying Lucy_Richardson deconvolution, we learn the to translate from the input image to the LR deconvolved image,
    this let's us run LR for a high number of iterations and control the noise by learning to deconvolve.
    """

    def __init__(
        self,
        psf_kernel,
        *args,
        deconvolution_class=ImageTranslatorLRDeconvScipy,
        deconvolving_translator=None,
        denoising_translator=None,
        feature_generator_class=StandardFeatureGenerator,
        regressor_class=CBRegressor,
        **kwargs,
    ):
        """Constructs a Lucy Richardson deconvolution image translator.

        Parameters
        ----------
        psf_kernel
            2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        args
        deconvolution_class
        deconvolving_translator
        denoising_translator
        feature_generator_class
        regressor_class
        kwargs
        """
        super().__init__(psf_kernel=psf_kernel)

        if deconvolving_translator is None:
            generator = feature_generator_class(
                include_spatial_features=True,
                include_scale_one=True,
                include_corner_features=True,
                include_fine_features=True,
            )
            regressor = regressor_class()

            deconvolving_translator = ImageTranslatorFGR(
                feature_generator=generator, regressor=regressor
            )

        if denoising_translator is None:
            generator = feature_generator_class(
                include_spatial_features=True,
                include_scale_one=True,
                include_corner_features=True,
                include_fine_features=True,
            )
            regressor = regressor_class()

            denoising_translator = ImageTranslatorFGR(
                feature_generator=generator, regressor=regressor
            )

        self.deconvolution_translator = deconvolution_class(psf_kernel, *args, **kwargs)
        self.learned_deconvolving_translator = deconvolving_translator
        self.learned_denoising_translator = denoising_translator

    def get_receptive_field_radius(self, ndim):
        """Returns the receptive field radius

        Parameters
        ----------
        ndim : int

        Returns
        -------
        int
            Receptive field radius

        """
        return self.deconvolution_translator.get_receptive_field_radius(ndim)

    def _train(self, *args, **kwargs):
        self.deconvolution_translator._train(*args, **kwargs)

    def _translate(self, *args, **kwargs):
        self.deconvolution_translator._translate(*args, **kwargs)

    def train(
        self,
        input_image,
        target_image=None,
        batch_axes=None,
        channel_axes=None,
        train_valid_ratio=0.1,
        callback_period=3,
        jinv=True,
    ):
        """Train method

        Parameters
        ----------
        input_image
        target_image
        batch_axes
        channel_axes
        train_valid_ratio
        callback_period
        jinv

        """

        self.learned_denoising_translator.train(target_image)
        denoised_target_image = self.learned_denoising_translator.translate(
            target_image
        )

        deconvolved_denoised_target_image = self.deconvolve(
            denoised_target_image,
            batch_axes=batch_axes,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
            force_jinv=jinv,
        )

        self.learned_deconvolving_translator.train(
            input_image, deconvolved_denoised_target_image, jinv=False
        )

    def deconvolve(
        self,
        image,
        batch_axes=None,
        train_valid_ratio=0.1,
        callback_period=3,
        force_jinv=None,
    ):
        """Deconvolves the given image

        Parameters
        ----------
        image
        batch_axes
        train_valid_ratio
        callback_period
        force_jinv

        Returns
        -------
        numpy.typing.ArrayLike
            Deconvolved image

        """

        self.deconvolution_translator.train(
            image,
            batch_axes=batch_axes,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
            jinv=force_jinv,
        )

        deconvolved_image = self.deconvolution_translator.translate(image)
        return deconvolved_image

    def translate(
        self,
        input_image,
        translated_image=None,
        batch_axes=None,
        channel_axes=None,
        tile_size=None,
        min_margin=8,
        max_margin=None,
        denormalise_values=True,
        leave_as_float=False,
        clip=True,
    ):
        """Translates the given image

        Parameters
        ----------
        input_image
        translated_image
        batch_axes
        channel_axes
        tile_size
        min_margin
        max_margin
        denormalise_values
        leave_as_float
        clip

        Returns
        -------
        numpy.typing.ArrayLike
            Translated image

        """
        return self.learned_deconvolving_translator.translate(
            input_image, translated_image, batch_axes, tile_size, max_margin
        )
