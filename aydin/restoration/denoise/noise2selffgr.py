import importlib
import inspect
import os
import shutil
import numpy

from aydin import regression
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.it.base import ImageTranslatorBase
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import lsection


if os.getenv("BUNDLED_AYDIN") == "1":
    from aydin.regression.lgbm import LGBMRegressor  # noqa: F401
    from aydin.regression.linear import LinearRegressor  # noqa: F401
    from aydin.regression.nn import NNRegressor  # noqa: F401
    from aydin.regression.random_forest import RandomForestRegressor  # noqa: F401
    from aydin.regression.support_vector import SupportVectorRegressor  # noqa: F401


class Noise2SelfFGR(DenoiseRestorationBase):
    """
    Noise2Self image denoising "Feature Generation & Regression" (FGR)
    """

    def __init__(
        self,
        *,
        variant: str = 'fgr-cb',
        use_model=None,
        input_model_path=None,
        lower_level_args=None,
        it_transforms=None,
    ):
        """
        Parameters
        ----------
        variant : str
            Variant of N2S FGR denoising
        use_model : bool
            Flag to choose to train a new model or infer from a
            previously trained model. By default it is None.
        input_model_path : str
            Path to model that is desired to be used for inference.
            By default it is None.
        lower_level_args : args
            Additional 'low-level' arguments to be passed.
        it_transforms :
            Transforms to be applied.
        """

        super().__init__()
        self.lower_level_args = lower_level_args

        self.backend_it, self.backend_regressor = (
            variant.split("-") if variant is not None else ("fgr", "cb")
        )

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

        # Feature Generator
        feature_generator = StandardFeatureGenerator
        fullargspec2 = inspect.getfullargspec(feature_generator.__init__)

        feature_generator_args = {
            "arguments": fullargspec2.args[4:],
            "defaults": fullargspec2.defaults[3:],
            "annotations": fullargspec2.annotations,
            "reference_class": feature_generator,
        }

        # IT FGR
        it = ImageTranslatorFGR

        fullargspec3 = inspect.getfullargspec(ImageTranslatorFGR.__init__)

        it_args = {
            "arguments": fullargspec3.args[3:],
            "defaults": fullargspec3.defaults[2:],
            "annotations": fullargspec3.annotations,
            "reference_class": it,
        }

        # Regressor
        regression_modules = DenoiseRestorationBase.get_implementations_in_a_module(
            regression
        )

        for module in regression_modules:
            regressor_args = self.get_class_implementation_kwonlyargs(
                regression, module, module.name.replace("_", "") + "Regressor"
            )

            arguments["Noise2SelfFGR-" + module.name] = {
                "feature_generator": feature_generator_args,
                "regressor": regressor_args,
                "it": it_args,
            }

        return arguments

    @property
    def implementations(self):
        """Returns the list of discovered implementations for given method."""
        return [
            "Noise2SelfFGR-" + x.name
            for x in self.get_implementations_in_a_module(regression)
        ]

    @property
    def implementations_description(self):
        fgr_description = Noise2SelfFGR.__doc__.strip()
        feature_generator_description = StandardFeatureGenerator.__doc__.strip()

        descriptions = []

        for module in self.get_implementations_in_a_module(regression):
            response = importlib.import_module(regression.__name__ + '.' + module.name)
            elem = [
                x
                for x in dir(response)
                if (module.name.replace("_", "") + "Regressor").lower() in x.lower()
            ][
                0
            ]  # class name

            elem_class = response.__getattribute__(elem)
            descriptions.append(
                fgr_description
                + ", uses "
                + feature_generator_description
                + " and uses "
                + elem_class.__doc__.replace("\n\n", "<br><br>")
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

    def get_generator(self):
        """Returns the corresponding generator instance for given selections.

        Returns
        -------
        generator : FeatureGeneratorBase

        """
        # print(self.lower_level_args)
        if self.lower_level_args is not None:
            generator = self.lower_level_args["feature_generator"]["class"](
                **self.lower_level_args["feature_generator"]["kwargs"]
            )
        else:
            generator = StandardFeatureGenerator()

        return generator

    def get_regressor(self):
        """Returns the corresponding regressor instance for given selections.

        Returns
        -------
        regressor : RegressorBase

        """

        if self.lower_level_args is not None:
            regressor = self.lower_level_args["regressor"]["class"](
                **self.lower_level_args["regressor"]["kwargs"]
            )
        else:
            regressor = CBRegressor()

        return regressor

    def get_translator(self, feature_generator, regressor):
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
            it = ImageTranslatorFGR(
                feature_generator=feature_generator,
                regressor=regressor,
                **self.lower_level_args["it"]["kwargs"]
                if self.lower_level_args is not None
                else {},
                blind_spots='auto',  # TODO: ACS: please set this as default upstream
            )

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

            self.it = self.get_translator(
                feature_generator=self.get_generator(), regressor=self.get_regressor()
            )

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


def noise2self_fgr(noisy, *, batch_axes=None, chan_axes=None, variant=None):
    """Method to denoise an image with trained Noise2Self FGR.

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
    # Run N2S and save the result
    n2s = Noise2SelfFGR(variant=variant)

    # Train
    n2s.train(noisy, batch_axes=batch_axes, chan_axes=chan_axes)

    # Denoise
    denoised = n2s.denoise(noisy, batch_axes=batch_axes, chan_axes=chan_axes)

    return denoised
