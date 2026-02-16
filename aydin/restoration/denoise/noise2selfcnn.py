"""Noise2Self Convolutional Neural Network (CNN) denoising module.

Provides the :class:`Noise2SelfCNN` denoising class and the convenience
function :func:`noise2self_cnn`. Uses self-supervised CNN architectures
(UNet, JiNet) for image denoising.
"""

import importlib
import inspect
import os
import shutil
from typing import Optional

import aydin.nn.models as models
from aydin.it.base import ImageTranslatorBase
from aydin.it.cnn_torch import ImageTranslatorCNNTorch
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import asection
from aydin.util.string.break_text import strip_notgui


class Noise2SelfCNN(DenoiseRestorationBase):
    """Noise2Self image denoising using Convolutional Neural Networks (CNN).

    Follows from the theory exposed in the <a
    href="https://arxiv.org/abs/1901.11365">Noise2Self paper</a>.
    Uses self-supervised blind-spot CNN architectures (UNet, JiNet) that
    learn to denoise images without requiring clean ground-truth data.
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
        """Construct a Noise2Self CNN denoiser.

        Parameters
        ----------
        variant : str, optional
            CNN architecture variant to use. Currently supported variants
            are ``'unet'`` and ``'jinet'``. Supersedes any variant specified
            in ``lower_level_args``.
        use_model : bool, optional
            If ``True``, load and use a previously trained model instead of
            training a new one.
        input_model_path : str, optional
            Path to a saved model zip file for inference.
        lower_level_args : dict, optional
            Additional low-level arguments passed to the underlying
            image translator and model constructors.
        it_transforms : list of dict, optional
            Custom list of transforms to apply. Each entry should have
            keys ``'class'`` and ``'kwargs'``. Defaults to Range, Padding,
            and Variance Stabilisation transforms.
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

        Discovers all available CNN model architectures and extracts their
        constructor signatures, paired with the :class:`ImageTranslatorCNNTorch`
        constructor arguments.

        Returns
        -------
        dict
            A nested dictionary keyed by implementation name (e.g.
            ``'Noise2SelfCNN-unet'``). Each value contains ``'model'``
            and ``'it'`` sub-dictionaries with their arguments, defaults,
            annotations, and reference class.
        """

        # IT CNN
        it = ImageTranslatorCNNTorch

        fullargspec3 = inspect.getfullargspec(ImageTranslatorCNNTorch.__init__)

        it_args = {
            "arguments": fullargspec3.args[1:],
            "defaults": fullargspec3.defaults or (),
            "annotations": fullargspec3.annotations,
            "reference_class": it,
        }

        # Model
        model_modules = DenoiseRestorationBase.get_implementations_in_a_module(models)

        arguments = {}

        for module in model_modules:
            # Find the actual model class name (e.g. ResidualUNetModel)
            # since module filenames don't always match class names directly
            response = importlib.import_module(models.__name__ + '.' + module.name)
            class_name = [
                x
                for x in dir(response)
                if x.endswith('Model') and not x.startswith('_')
            ][0]
            model_args = self.get_class_implementation_kwargs(
                models, module, class_name
            )
            arguments["Noise2SelfCNN-" + module.name] = {
                "model": model_args,
                "it": it_args,
            }

        return arguments

    @property
    def implementations(self):
        """Return the list of available CNN implementation names.

        Discovers all CNN model modules and returns their names prefixed
        with ``'Noise2SelfCNN-'``.

        Returns
        -------
        list of str
            Implementation variant names (e.g. ``['Noise2SelfCNN-unet',
            'Noise2SelfCNN-jinet']``).
        """
        return [
            "Noise2SelfCNN-" + x.name
            for x in self.get_implementations_in_a_module(models)
        ]

    @property
    def implementations_description(self):
        """Return human-readable descriptions for each CNN implementation.

        Builds a description for each variant by combining the
        :class:`Noise2SelfCNN` docstring with the individual model
        class docstring.

        Returns
        -------
        list of str
            HTML-formatted description strings, one per implementation,
            in the same order as :attr:`implementations`.
        """
        cnn_description = strip_notgui(Noise2SelfCNN.__doc__.strip())

        descriptions = []

        for module in self.get_implementations_in_a_module(models):
            response = importlib.import_module(models.__name__ + '.' + module.name)
            # Find the model class by looking for a class ending with "Model"
            # This handles naming mismatches like res_unet -> ResidualUNetModel
            elem = [
                x
                for x in dir(response)
                if x.endswith('Model') and not x.startswith('_')
            ][0]

            elem_class = getattr(response, elem)
            # model_name = elem_class.__name__
            model_description = strip_notgui(elem_class.__doc__).replace(
                "\n\n", "<br><br>"
            )

            descriptions.append(cnn_description + f"<br><br>{model_description}")

            # elem_class = response.__getattribute__(elem)

        return descriptions

    def stop_running(self):
        """Stop the current Noise2Self CNN training or inference.

        Delegates to the underlying image translator's ``stop_training``
        method to halt the training loop.
        """
        self.it.stop_training()

    def get_translator(self):
        """Return the image translator for the current configuration.

        If ``variant`` is set, creates a new :class:`ImageTranslatorCNNTorch`
        with the specified model architecture. If ``use_model_flag`` is
        set, loads a pre-trained model from disk. Otherwise, creates a
        new translator using ``lower_level_args``.

        Returns
        -------
        ImageTranslatorBase
            Configured image translator instance ready for training or
            inference.
        """
        if self.variant:
            return ImageTranslatorCNNTorch(model=self.variant)

        # Use a pre-saved model or train a new one from scratch and save it
        if self.use_model_flag:
            # Unarchive the model file and load its ImageTranslator object into self.it
            shutil.unpack_archive(
                self.input_model_path, os.path.dirname(self.input_model_path), "zip"
            )
            it = ImageTranslatorBase.load(self.input_model_path[:-4])
        else:
            it = ImageTranslatorCNNTorch(
                **(
                    self.lower_level_args["it"]["kwargs"]
                    if self.lower_level_args is not None
                    else {}
                )
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
        """Train the Noise2Self CNN denoiser on a noisy image.

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
            if channel_axes is not None and len(channel_axes) > 0 and any(channel_axes):
                pass  # Channel axes provided, continue with training

            self.it = self.get_translator()

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
        """Denoise an image using the trained Noise2Self CNN model.

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


def noise2self_cnn(image, *, batch_axes=None, channel_axes=None, variant=None):
    """Denoise an image using Noise2Self CNN in a single call.

    Convenience function that creates a :class:`Noise2SelfCNN` instance,
    trains it on the noisy image, and returns the denoised result.

    Parameters
    ----------
    image : numpy.ndarray
        Image to denoise.
    batch_axes : array_like, optional
        Indices of batch axes.
    channel_axes : array_like, optional
        Indices of channel axes.
    variant : str, optional
        CNN architecture variant. Available variants: ``'unet'``,
        ``'jinet'``. When ``None``, the default architecture is used.

    Returns
    -------
    numpy.ndarray
        Denoised image.

    """
    # Run N2S and save the result
    n2s = Noise2SelfCNN(variant=variant)

    # Train
    n2s.train(image, batch_axes=batch_axes, channel_axes=channel_axes)

    # Denoise
    denoised = n2s.denoise(image, batch_axes=batch_axes, channel_axes=channel_axes)

    return denoised
