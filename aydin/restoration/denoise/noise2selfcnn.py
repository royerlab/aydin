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

from aydin.it.base import ImageTranslatorBase

# from aydin.it.cnn_torch import ImageTranslatorCNNTorch
from aydin.it.cnn import ImageTranslatorCNN
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.nn.tf import models
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import asection


class Noise2SelfCNN(DenoiseRestorationBase):
    """Noise2Self image denoising using Convolutional Neural Networks (CNN).

    Follows from the theory exposed in the
    `Noise2Self paper <https://arxiv.org/abs/1901.11365>`_.
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
        """Returns the configurable arguments that will be exposed
        on GUI and CLI.
        """

        # IT CNN
        it = ImageTranslatorCNN

        fullargspec3 = inspect.getfullargspec(ImageTranslatorCNN.__init__)

        it_args = {
            "arguments": fullargspec3.args[1:],
            "defaults": fullargspec3.defaults,
            "annotations": fullargspec3.annotations,
            "reference_class": it,
        }

        # Model
        model_modules = DenoiseRestorationBase.get_implementations_in_a_module(models)

        arguments = {}

        for module in model_modules:
            model_args = self.get_class_implementation_kwargs(
                models, module, module.name + "Model"
            )
            arguments["Noise2SelfCNN-" + module.name] = {
                "model": model_args,
                "it": it_args,
            }

        return arguments

    @property
    def implementations(self):
        """Returns the list of discovered implementations for given method."""
        return [
            "Noise2SelfCNN-" + x.name
            for x in self.get_implementations_in_a_module(models)
        ]

    @property
    def implementations_description(self):
        """Return human-readable descriptions for each CNN implementation."""
        cnn_description = Noise2SelfCNN.__doc__.strip()

        descriptions = []

        for module in self.get_implementations_in_a_module(models):
            response = importlib.import_module(models.__name__ + '.' + module.name)
            elem = [
                x for x in dir(response) if module.name.replace("_", "") in x.lower()
            ][
                0
            ]  # class name

            elem_class = response.__getattribute__(elem)
            # model_name = elem_class.__name__
            model_description = elem_class.__doc__.replace("\n\n", "<br><br>")

            descriptions.append(cnn_description + f"<br><br>{model_description}")

            # elem_class = response.__getattribute__(elem)

        return descriptions

    def stop_running(self):
        """Stop the current Noise2Self CNN training or inference."""
        self.it.stop_training()

    def get_translator(self):
        """Returns the corresponding translator instance for given selections.

        Returns
        -------
        it : ImageTranslatorBase

        """
        if self.variant:
            return ImageTranslatorCNN(model_architecture=self.variant)

        # Use a pre-saved model or train a new one from scratch and save it
        if self.use_model_flag:
            # Unarchive the model file and load its ImageTranslator object into self.it
            shutil.unpack_archive(
                self.input_model_path, os.path.dirname(self.input_model_path), "zip"
            )
            it = ImageTranslatorBase.load(self.input_model_path[:-4])
        else:
            it = ImageTranslatorCNN(
                **(
                    self.lower_level_args["it"]["kwargs"]
                    if self.lower_level_args is not None
                    else {}
                )
            )

        return it

    def add_transforms(self):
        """Add the configured image transforms to the image translator."""
        if self.it_transforms is not None:
            for transform in self.it_transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                self.it.add_transform(transform_class(**transform_kwargs))

    def train(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Train the Noise2Self CNN denoiser on a noisy image.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image.
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments. Supports ``'train_valid_ratio'``,
            ``'callback_period'``, and ``'jinv'``.
        """
        with asection("Noise2Self train is starting..."):
            if chan_axes is not None and len(chan_axes) > 0 and any(chan_axes):
                pass  # Channel axes provided, continue with training

            self.it = self.get_translator()

            self.add_transforms()

            # Train a new model
            self.it.train(
                noisy_image,
                noisy_image,
                batch_axes=batch_axes,
                channel_axes=chan_axes,
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

    def denoise(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Denoise an image using the trained Noise2Self CNN model.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to denoise.
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
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
                channel_axes=chan_axes,
                tile_size=kwargs['tile_size'] if 'tile_size' in kwargs else None,
            )

            response = response.astype(noisy_image.dtype, copy=False)

            return response


def noise2self_cnn(image, *, batch_axes=None, chan_axes=None, variant=None):
    """Denoise an image using Noise2Self CNN in a single call.

    Convenience function that creates a :class:`Noise2SelfCNN` instance,
    trains it on the noisy image, and returns the denoised result.

    Parameters
    ----------
    image : numpy.ndarray
        Image to denoise.
    batch_axes : array_like, optional
        Indices of batch axes.
    chan_axes : array_like, optional
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
    n2s.train(image, batch_axes=batch_axes, chan_axes=chan_axes)

    # Denoise
    denoised = n2s.denoise(image, batch_axes=batch_axes, chan_axes=chan_axes)

    return denoised
