"""Noise2Self Feature Generation and Regression (FGR) denoising module.

Provides the :class:`Noise2SelfFGR` denoising class and the convenience
function :func:`noise2self_fgr`. This approach combines feature generation
with gradient boosting or other regression methods for self-supervised
image denoising.
"""

import importlib
import inspect
import os
import shutil
from typing import Optional

from aydin import regression
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.it.base import ImageTranslatorBase
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import aprint, asection
from aydin.util.string.break_text import strip_notgui


class Noise2SelfFGR(DenoiseRestorationBase):
    """Noise2Self image denoising using Feature Generation & Regression (FGR).

    Follows from the theory exposed in the <a
    href="https://arxiv.org/abs/1901.11365">Noise2Self paper</a>.
    Combines feature extraction (spatial, frequency-domain, correlation-based)
    with gradient boosting or other regression methods to predict clean pixel
    values in a self-supervised manner.
    <notgui>
    """

    def __init__(
        self,
        *,
        variant: Optional[str] = None,
        use_model=None,
        input_model_path=None,
        lower_level_args=None,
        it_transforms=None,
    ):
        """Construct a Noise2Self FGR denoiser.

        Parameters
        ----------
        variant : str, optional
            Regression algorithm variant to use. Supersedes the regressor
            option in ``lower_level_args``. The :attr:`implementations`
            property returns a complete list of available variants (prefixed
            with ``'Noise2SelfFGR-'``). Example variants: ``'cb'``,
            ``'lgbm'``, ``'linear'``, ``'perceptron'``, ``'random_forest'``,
            ``'support_vector'``.
        use_model : bool, optional
            If ``True``, load and use a previously trained model instead of
            training a new one.
        input_model_path : str, optional
            Path to a saved model zip file for inference.
        lower_level_args : dict, optional
            Additional low-level arguments passed to the underlying feature
            generator, regressor, and image translator constructors. Expected
            keys include ``'feature_generator'``, ``'regressor'``, and
            ``'it'``, each containing ``'class'`` and ``'kwargs'`` entries.
        it_transforms : list of dict, optional
            Custom list of transforms to apply. Each entry should have keys
            ``'class'`` and ``'kwargs'``. Defaults to Range, Padding, and
            Variance Stabilisation transforms.
        """
        super().__init__(variant=variant)
        self.use_model_flag = use_model
        self.input_model_path = input_model_path
        self.lower_level_args = lower_level_args

        self.it_transforms = (
            [
                {"class": RangeTransform, "kwargs": {}},
                {"class": PaddingTransform, "kwargs": {}},
                {"class": VarianceStabilisationTransform, "kwargs": {}},
            ]
            if it_transforms is None
            else it_transforms
        )

    @property
    def configurable_arguments(self):
        """Return configurable arguments for the GUI and CLI.

        Discovers all available regressor modules, extracts their constructor
        signatures, and pairs them with the :class:`StandardFeatureGenerator`
        and :class:`ImageTranslatorFGR` constructor arguments.

        Returns
        -------
        dict
            A nested dictionary keyed by implementation name (e.g.
            ``'Noise2SelfFGR-cb'``). Each value contains
            ``'feature_generator'``, ``'regressor'``, and ``'it'``
            sub-dictionaries with their arguments, defaults, annotations,
            and reference class.
        """
        arguments = {}

        # Feature Generator
        feature_generator = StandardFeatureGenerator
        fullargspec2 = inspect.getfullargspec(feature_generator.__init__)

        feature_generator_args = {
            "arguments": fullargspec2.args[4:],
            "defaults": (fullargspec2.defaults or ())[3:],
            "annotations": fullargspec2.annotations,
            "reference_class": feature_generator,
        }

        # IT FGR
        it = ImageTranslatorFGR

        fullargspec3 = inspect.getfullargspec(ImageTranslatorFGR.__init__)

        it_args = {
            "arguments": fullargspec3.args[3:],
            "defaults": (fullargspec3.defaults or ())[2:],
            "annotations": fullargspec3.annotations,
            "reference_class": it,
        }

        # Regressor
        regression_modules = DenoiseRestorationBase.get_implementations_in_a_module(
            regression
        )

        for module in regression_modules:
            try:
                regressor_args = self.get_class_implementation_kwargs(
                    regression, module, module.name.replace("_", "") + "Regressor"
                )

                arguments["Noise2SelfFGR-" + module.name] = {
                    "feature_generator": feature_generator_args,
                    "regressor": regressor_args,
                    "it": it_args,
                }
            except Exception:
                continue

        return arguments

    @property
    def implementations(self):
        """Return the list of available FGR implementation names.

        Discovers all regressor modules and returns their names prefixed
        with ``'Noise2SelfFGR-'``.

        Returns
        -------
        list of str
            Implementation variant names (e.g. ``['Noise2SelfFGR-cb',
            'Noise2SelfFGR-lgbm', ...]``).
        """
        result = []
        for x in self.get_implementations_in_a_module(regression):
            try:
                importlib.import_module(regression.__name__ + '.' + x.name)
                result.append("Noise2SelfFGR-" + x.name)
            except Exception:
                continue
        return result

    @property
    def implementations_description(self):
        """Return human-readable descriptions for each FGR implementation.

        Builds a description for each variant by combining the
        :class:`Noise2SelfFGR` docstring, the feature generator
        description, and the individual regressor class docstring.

        Returns
        -------
        list of str
            HTML-formatted description strings, one per implementation,
            in the same order as :attr:`implementations`.
        """
        fgr_description = strip_notgui(Noise2SelfFGR.__doc__.strip())

        feature_generator_name = StandardFeatureGenerator.__name__.replace(
            "FeatureGenerator", ""
        )
        feature_generator_description = strip_notgui(
            StandardFeatureGenerator.__doc__.strip()
        )

        descriptions = []

        for module in self.get_implementations_in_a_module(regression):
            try:
                response = importlib.import_module(
                    regression.__name__ + '.' + module.name
                )
                elem = [
                    x
                    for x in dir(response)
                    if (module.name.replace("_", "") + "Regressor").lower() in x.lower()
                ][
                    0
                ]  # class name

                elem_class = response.__getattribute__(elem)
                regressor_name = elem_class.__name__.replace("Regressor", "")
                regressor_description = strip_notgui(elem_class.__doc__).replace(
                    "\n\n", "<br><br>"
                )

                descriptions.append(
                    fgr_description
                    + f" Uses the {feature_generator_name}"
                    + " feature generator and"
                    + f" {regressor_name} regressor. "
                    + f"<br><br>About the feature generator: {feature_generator_description}"  # noqa: E501
                    + "<br><br>About the regressor: "
                    + regressor_description
                )
            except Exception:
                continue

        return descriptions

    def stop_running(self):
        """Stop the current Noise2Self FGR training or inference.

        Delegates to the underlying image translator's ``stop_training``
        method to halt the training process.
        """
        self.it.stop_training()

    def get_generator(self):
        """Return the feature generator for the current configuration.

        If ``lower_level_args`` provides a feature generator class and
        keyword arguments, those are used. Otherwise, a default
        :class:`StandardFeatureGenerator` is created.

        Returns
        -------
        FeatureGeneratorBase
            Configured feature generator instance.
        """
        if self.lower_level_args is not None:
            generator = self.lower_level_args["feature_generator"]["class"](
                **self.lower_level_args["feature_generator"]["kwargs"]
            )
        else:
            generator = StandardFeatureGenerator()

        return generator

    def get_regressor(self):
        """Return the regressor for the current configuration.

        If ``variant`` is set, returns the corresponding regressor class
        instance (e.g. :class:`CBRegressor` for ``'cb'``). If
        ``lower_level_args`` provides a regressor class and keyword
        arguments, those are used. Otherwise, defaults to
        :class:`CBRegressor`.

        Returns
        -------
        RegressorBase
            Configured regressor instance.
        """
        if self.variant:
            regressor_modules = {
                "cb": ("aydin.regression.cb", "CBRegressor"),
                "lgbm": ("aydin.regression.lgbm", "LGBMRegressor"),
                "linear": ("aydin.regression.linear", "LinearRegressor"),
                "perceptron": ("aydin.regression.perceptron", "PerceptronRegressor"),
                "random_forest": (
                    "aydin.regression.random_forest",
                    "RandomForestRegressor",
                ),
                "support_vector": (
                    "aydin.regression.support_vector",
                    "SupportVectorRegressor",
                ),
            }
            module_path, class_name = regressor_modules[self.variant]
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)()

        if self.lower_level_args is None:
            try:
                from aydin.regression.cb import CBRegressor

                regressor = CBRegressor()
            except ImportError:
                aprint(
                    "Warning: CatBoost is not installed"
                    " — falling back to LightGBM"
                    " regressor. For best results,"
                    " install CatBoost with:"
                    " pip install catboost"
                )
                from aydin.regression.lgbm import LGBMRegressor

                regressor = LGBMRegressor()
        else:
            regressor = self.lower_level_args["regressor"]["class"](
                **self.lower_level_args["regressor"]["kwargs"]
            )

        return regressor

    def get_translator(self, feature_generator, regressor):
        """Return the image translator for the current configuration.

        If ``use_model_flag`` is set, loads a pre-trained model from disk.
        Otherwise, creates a new :class:`ImageTranslatorFGR` with the
        provided feature generator and regressor.

        Parameters
        ----------
        feature_generator : FeatureGeneratorBase
            The feature generator to use for extracting features from images.
        regressor : RegressorBase
            The regressor to use for mapping features to denoised values.

        Returns
        -------
        ImageTranslatorBase
            Configured image translator instance ready for training or
            inference.
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
                **(
                    self.lower_level_args["it"]["kwargs"]
                    if self.lower_level_args is not None
                    else {}
                ),
            )

        return it

    def add_transforms(self):
        """Add the configured image transforms to the image translator.

        Iterates over ``self.it_transforms`` and adds each transform
        instance to ``self.it``. Transforms are applied in order during
        training and inference (e.g. range normalisation, padding,
        variance stabilisation).
        """
        if self.it_transforms is not None:
            for transform in self.it_transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                self.it.add_transform(transform_class(**transform_kwargs))

    def train(self, noisy_image, *, batch_axes=None, channel_axes=None, **kwargs):
        """Train the Noise2Self FGR denoiser on a noisy image.

        Generates features from the noisy image and trains a regression
        model to predict clean pixel values in a self-supervised manner.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image.
        batch_axes : array_like, optional
            Indices of batch axes.
        channel_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments. Supports ``'train_valid_ratio'``,
            ``'callback_period'``, and ``'jinv'``.
        """
        with asection("Noise2Self train is starting..."):

            self.it = self.get_translator(
                feature_generator=self.get_generator(), regressor=self.get_regressor()
            )

            self.add_transforms()

            # Train a new model
            self.it.train(
                noisy_image,
                noisy_image,
                batch_axes=batch_axes,
                channel_axes=channel_axes,
                train_valid_ratio=(
                    kwargs['train_valid_ratio']
                    if 'train_valid_ratio' in kwargs
                    else 0.1
                ),
                callback_period=(
                    kwargs['callback_period'] if 'callback_period' in kwargs else 3
                ),
                jinv=kwargs['jinv'] if 'jinv' in kwargs else None,
            )

    def denoise(self, noisy_image, *, batch_axes=None, channel_axes=None, **kwargs):
        """Denoise an image using the trained Noise2Self FGR model.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to denoise.
        batch_axes : array_like, optional
            Indices of batch axes.
        channel_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments. Supports ``'tile_size'``.

        Returns
        -------
        numpy.ndarray
            The denoised image, cast to the input image's dtype.
        """
        with asection("Noise2Self denoise is starting..."):

            # Predict the resulting image
            response = self.it.translate(
                noisy_image,
                batch_axes=batch_axes,
                channel_axes=channel_axes,
                tile_size=kwargs['tile_size'] if 'tile_size' in kwargs else None,
            )

            response = response.astype(noisy_image.dtype, copy=False)

            return response


def noise2self_fgr(noisy, *, batch_axes=None, channel_axes=None, variant=None):
    """Denoise an image using Noise2Self FGR in a single call.

    Convenience function that creates a :class:`Noise2SelfFGR` instance,
    trains it on the noisy image, and returns the denoised result.

    Parameters
    ----------
    noisy : numpy.ndarray
        Image to denoise.
    batch_axes : array_like, optional
        Indices of batch axes.
    channel_axes : array_like, optional
        Indices of channel axes.
    variant : str, optional
        Regression algorithm variant. Available variants: ``'cb'``
        (CatBoost, default), ``'lgbm'`` (LightGBM), ``'linear'``,
        ``'perceptron'``, ``'random_forest'``, ``'support_vector'``.
        When ``None``, CatBoost is used.

    Returns
    -------
    numpy.ndarray
        Denoised image.

    """
    # Run N2S and save the result
    n2s = Noise2SelfFGR(variant=variant)

    # Train
    n2s.train(noisy, batch_axes=batch_axes, channel_axes=channel_axes)

    # Denoise
    denoised = n2s.denoise(noisy, batch_axes=batch_axes, channel_axes=channel_axes)

    return denoised
