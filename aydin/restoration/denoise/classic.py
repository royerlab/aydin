import importlib
import os
import platform
import shutil
import numpy

from aydin.it import classic_denoisers
from aydin.it.base import ImageTranslatorBase
from aydin.it.classic import ImageDenoiserClassic
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import lsection


if os.getenv("BUNDLED_AYDIN") == "1":
    from aydin.it.classic_denoisers.bilateral import (  # noqa: F401
        calibrate_denoise_bilateral,
        denoise_bilateral,
    )
    from aydin.it.classic_denoisers.bmnd import (  # noqa: F401
        calibrate_denoise_bmnd,
        denoise_bmnd,
    )
    from aydin.it.classic_denoisers.butterworth import (  # noqa: F401
        calibrate_denoise_butterworth,
        denoise_butterworth,
    )
    from aydin.it.classic_denoisers.dictionary_fixed import (  # noqa: F401
        calibrate_denoise_dictionary_fixed,
        denoise_dictionary_fixed,
    )
    from aydin.it.classic_denoisers.dictionary_learned import (  # noqa: F401
        calibrate_denoise_dictionary_learned,
        denoise_dictionary_learned,
    )
    from aydin.it.classic_denoisers.gaussian import (  # noqa: F401
        calibrate_denoise_gaussian,
        denoise_gaussian,
    )
    from aydin.it.classic_denoisers.gm import (  # noqa: F401
        calibrate_denoise_gm,
        denoise_gm,
    )
    from aydin.it.classic_denoisers.harmonic import (  # noqa: F401
        calibrate_denoise_harmonic,
        denoise_harmonic,
    )
    from aydin.it.classic_denoisers.lipschitz import (  # noqa: F401
        calibrate_denoise_lipschitz,
        denoise_lipschitz,
    )
    from aydin.it.classic_denoisers.nlm import (  # noqa: F401
        calibrate_denoise_nlm,
        denoise_nlm,
    )
    from aydin.it.classic_denoisers.pca import (  # noqa: F401
        calibrate_denoise_pca,
        denoise_pca,
    )
    from aydin.it.classic_denoisers.spectral import (  # noqa: F401
        calibrate_denoise_spectral,
        denoise_spectral,
    )
    from aydin.it.classic_denoisers.tv import (  # noqa: F401
        calibrate_denoise_tv,
        denoise_tv,
    )
    from aydin.it.classic_denoisers.wavelet import (  # noqa: F401
        calibrate_denoise_wavelet,
        denoise_wavelet,
    )


class Classic(DenoiseRestorationBase):
    """Classic Image Denoising.

    Parameters
    ----------
    variant : str
        Opacity of the layer visual, between 0.0 and 1.0.
    use_model : bool
        Flag to choose to train a new model or infer from a
        previously trained model. By default it is None.
    input_model_path : string
        Path to model that is desired to be used for inference.
        By default it is None.
    """

    disabled_modules = ["bilateral", "bmnd"]

    def __init__(
        self,
        *,
        variant: str = 'butterworth',
        use_model=None,
        input_model_path=None,
        lower_level_args=None,
        it_transforms=None,
    ):
        super().__init__()
        self.lower_level_args = lower_level_args

        self.backend = variant

        self.input_model_path = input_model_path
        self.use_model_flag = use_model
        self.model_folder_path = None

        self.it = None
        self.it_transforms = (
            [
                {"class": RangeTransform, "kwargs": {}},
                {"class": PaddingTransform, "kwargs": {}},
                {"class": VarianceStabilisationTransform, "kwargs": {}},
            ]
            if it_transforms is None
            else it_transforms
        )

        self.has_less_than_one_million_voxels = False
        self.has_less_than_one_trillion_voxels = True
        self.number_of_dims = -1

    @property
    def configurable_arguments(self):
        """Returns the configurable arguments that will be exposed
        on GUI and CLI.
        """
        arguments = {}

        # Methods
        method_modules = self.get_implementations_in_a_module(classic_denoisers)

        # Remove the disabled modules
        method_modules = [
            module
            for module in method_modules
            if module.name not in self.disabled_modules
        ]

        if platform.system() == "Darwin":
            for module in method_modules:
                if module.name in ["spectral", "dictionary_learned"]:
                    method_modules.remove(module)

        for module in method_modules:
            calibration_args = self.get_function_implementation_kwonlyargs(
                classic_denoisers, module, "calibrate_denoise_" + module.name
            )
            for idx, arg_name in enumerate(calibration_args["arguments"]):
                if arg_name == "display_images":
                    calibration_args["arguments"].remove(arg_name)
                    calibration_args["defaults"] = tuple(
                        x
                        for id, x in enumerate(calibration_args["defaults"])
                        if id != idx
                    )
                    del calibration_args["annotations"][arg_name]

            calibration_args["backend"] = module.name
            arguments["Classic-" + module.name] = {"calibration": calibration_args}

        return arguments

    @property
    def implementations(self):
        """Returns the list of discovered implementations for given method."""
        method_modules = self.get_implementations_in_a_module(classic_denoisers)

        # Remove the disabled modules
        method_modules = [
            module
            for module in method_modules
            if module.name not in self.disabled_modules
        ]

        if platform.system() == "Darwin":
            for module in method_modules:
                if module.name in ["spectral", "dictionary_learned"]:
                    method_modules.remove(module)

        return ["Classic-" + x.name for x in method_modules]

    @property
    def implementations_description(self):
        it_classic_description = ImageDenoiserClassic.__doc__.strip()

        descriptions = []

        method_modules = self.get_implementations_in_a_module(classic_denoisers)

        # Remove the disabled modules
        method_modules = [
            module
            for module in method_modules
            if module.name not in self.disabled_modules
        ]

        if platform.system() == "Darwin":
            for module in method_modules:
                if module.name in ["spectral", "dictionary_learned"]:
                    method_modules.remove(module)

        for module in method_modules:
            response = importlib.import_module(
                classic_denoisers.__name__ + '.' + module.name
            )

            elem = response.__getattribute__("denoise_" + module.name)
            descriptions.append(
                it_classic_description
                + ": "
                + module.name
                + "\n\n"
                + elem.__doc__[: elem.__doc__.find("Parameters")].replace(
                    "\n\n", "<br><br>"
                )
            )

        return descriptions

    def stop_running(self):
        """Method to stop running N2S instance"""
        self.it.stop_training()

    def set_image_metrics(self, image_shape):
        """Sets several image metric parameters used internally.

        Parameters
        ----------
        image_shape : tuple

        """
        self.number_of_dims = len(image_shape)
        number_of_voxels = numpy.prod(numpy.array(image_shape))
        self.has_less_than_one_million_voxels = number_of_voxels < 1000000
        self.has_less_than_one_trillion_voxels = number_of_voxels < 1000000000000

    def get_translator(self):
        """Returns the corresponding translator instance for given selections.

        Parameters
        ----------
        feature_generator : FeatureGeneratorBase
        regressor : RegressorBase

        Returns
        -------
        it : ImageTranslatorBase

        """
        # Use a pre-saved model or train a new one from scratch and save it
        if self.use_model_flag:
            # Unarchive the model file and load its ImageTranslator object into self.it
            shutil.unpack_archive(
                self.input_model_path, os.path.dirname(self.input_model_path), "zip"
            )
            it = ImageTranslatorBase.load(self.input_model_path[:-4])
        else:
            if self.lower_level_args is not None:
                method = (
                    self.backend
                    if self.lower_level_args["variant"] is None
                    else self.lower_level_args["variant"].split("-")[1]
                )
                it = ImageDenoiserClassic(
                    method=method,
                    calibration_kwargs=self.lower_level_args["calibration"]["kwargs"],
                )
            else:
                it = ImageDenoiserClassic(method=self.backend)

        return it

    def add_transforms(self):
        if self.it_transforms is not None:
            for transform in self.it_transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                self.it.add_transform(transform_class(**transform_kwargs))

    def train(
        self, noisy_image, *, batch_axes=None, chan_axes=None, image_path=None, **kwargs
    ):
        """Method to run training for Noise2Self FGR.

        Parameters
        ----------
        noisy_image : numpy.ndarray
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.
        image_path : str

        Returns
        -------
        response : numpy.ndarray

        """
        with lsection("Noise2Self train is starting..."):
            self.set_image_metrics(noisy_image.shape)

            self.it = self.get_translator()

            self.add_transforms()

            # Train a new model
            self.it.train(
                noisy_image,
                noisy_image,
                batch_axes=batch_axes,
                channel_axes=chan_axes,
                train_valid_ratio=kwargs['train_valid_ratio']
                if 'train_valid_ratio' in kwargs
                else 0.1,
                callback_period=kwargs['callback_period']
                if 'callback_period' in kwargs
                else 3,
                jinv=kwargs['jinv'] if 'jinv' in kwargs else None,
            )

            # Save the trained model
            # self.save_model(image_path)  # TODO:  fix the problems here

    def denoise(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Method to denoise an image with trained Noise2Self FGR.

        Parameters
        ----------
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.
        noisy_image : numpy.ndarray

        Returns
        -------
        response : numpy.ndarray

        """
        with lsection("Noise2Self denoise is starting..."):

            # Predict the resulting image
            response = self.it.translate(
                noisy_image,
                batch_axes=batch_axes,
                channel_axes=chan_axes,
                tile_size=kwargs['tile_size'] if 'tile_size' in kwargs else None,
            )

            response = response.astype(noisy_image.dtype, copy=False)

            return response


def classic_denoise(noisy, *, batch_axes=None, chan_axes=None, variant=None):
    """Method to denoise an image with Classic denoising restoration module.

    Parameters
    ----------
    noisy : numpy.ndarray
        Image to denoise
    batch_axes : array_like, optional
        Indices of batch axes.
    chan_axes : array_like, optional
        Indices of channel axes.
    variant : str
        Algorithm variant.

    Returns
    -------
    Denoised image : numpy.ndarray

    """
    # Run and save the result
    classic = Classic(variant=variant)

    # Train
    classic.train(noisy, batch_axes=batch_axes, chan_axes=chan_axes)

    # Denoise
    denoised = classic.denoise(noisy, batch_axes=batch_axes, chan_axes=chan_axes)

    return denoised
