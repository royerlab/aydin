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
from aydin.util.log.log import lsection


class Noise2SelfCNN(DenoiseRestorationBase):
    """
    Noise2Self image denoising using the "Convolutional Neural Networks" (
    CNN) approach. Follows from the theory exposed in the <a
    href="https://arxiv.org/abs/1901.11365">Noise2Self paper</a>.
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
        """
        Noise2Self image denoising using "Convolutional Neural Networks" (CNN).

        Parameters
        ----------
        variant : str, optional
            Variant of CNN denoiser to be used. Variant would supersede
            the denoiser option passed in lower_level_args. Currently, we
            support only two variants: `unet` and `jinet`.
        use_model : bool
            Flag to choose to train a new model or infer from a
            previously trained model. By default it is None.
        input_model_path : string
            Path to model that is desired to be used for inference.
            By default it is None.
        """
        super().__init__()
        self.variant = variant
        self.use_model_flag = use_model
        self.input_model_path = input_model_path
        self.lower_level_args = lower_level_args

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
        """Method to stop running N2S instance"""
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
                **self.lower_level_args["it"]["kwargs"]
                if self.lower_level_args is not None
                else {}
            )

        return it

    def add_transforms(self):
        if self.it_transforms is not None:
            for transform in self.it_transforms:
                transform_class = transform["class"]
                transform_kwargs = transform["kwargs"]
                self.it.add_transform(transform_class(**transform_kwargs))

    def train(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Method to run Noise2Self CNN training.

        Parameters
        ----------
        noisy_image : numpy.ArrayLike
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.

        Returns
        -------
        response : numpy.ArrayLike

        """
        with lsection("Noise2Self train is starting..."):
            if sum(chan_axes):
                return

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

    def denoise(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Method to denoise an image with trained Noise2Self.

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


def noise2self_cnn(image, *, batch_axes=None, chan_axes=None, variant=None):
    """Method to denoise an image with Noise2Self CNN.

    Parameters
    ----------
    image : numpy.ndarray
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
    n2s = Noise2SelfCNN(variant=variant)

    # Train
    n2s.train(image, batch_axes=batch_axes, chan_axes=chan_axes)

    # Denoise
    denoised = n2s.denoise(image, batch_axes=batch_axes, chan_axes=chan_axes)

    return denoised
