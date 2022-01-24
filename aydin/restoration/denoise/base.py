import importlib
import inspect
import os
import pkgutil
import shutil
from abc import abstractmethod, ABC
from pathlib import Path

from aydin.util.log.log import lprint


class DenoiseRestorationBase(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def configurable_arguments(self):
        """Returns the configurable arguments that will be exposed
        on GUI and CLI.
        """
        NotImplementedError()

    @property
    @abstractmethod
    def implementations(self):
        """Returns the list of discovered implementations for given method."""
        NotImplementedError()

    @abstractmethod
    def stop_running(self):
        """Method to stop running restoration instance."""
        NotImplementedError()

    @abstractmethod
    def train(
        self, noisy_image, *, batch_axes=None, chan_axes=None, image_path=None, **kwargs
    ):
        """Method to run training.

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
        NotImplementedError()

    @abstractmethod
    def denoise(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Method to denoise an image.

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
        NotImplementedError()

    @staticmethod
    def get_implementations_in_a_module(module):
        return [
            x
            for x in pkgutil.iter_modules(tuple(module.__path__))
            if not x.ispkg and x.name != 'base'
        ]

    @staticmethod
    def get_class_implementation_kwonlyargs(package, module, implementation_class_name):
        response = importlib.import_module(package.__name__ + '.' + module.name)
        elem = [
            x for x in dir(response) if implementation_class_name.lower() in x.lower()
        ][
            0
        ]  # class name

        class_itself = response.__getattribute__(elem)
        fullargspec = inspect.getfullargspec(class_itself.__init__)

        args_offset = len(fullargspec.args) - len(fullargspec.defaults)

        args = {
            "arguments": fullargspec.args[args_offset:],
            "defaults": fullargspec.defaults,
            "annotations": fullargspec.annotations,
            "reference_class": class_itself,
        }

        return args

    @staticmethod
    def get_function_implementation_kwonlyargs(
        package, module, implementation_function_name
    ):
        response = importlib.import_module(package.__name__ + '.' + module.name)

        function_itself = response.__getattribute__(implementation_function_name)

        fullargspec = inspect.getfullargspec(function_itself)

        args_offset = len(fullargspec.args) - len(fullargspec.defaults)

        args = {
            "arguments": fullargspec.args[args_offset:],
            "defaults": fullargspec.defaults,
            "annotations": fullargspec.annotations,
            "reference_class": function_itself,
        }

        return args

    @staticmethod
    def clean_model_folder(model_folder):
        """Method to clean model folder created"""
        shutil.rmtree(model_folder)

    @staticmethod
    def archive_model(source, destination):
        """Archives the model to given destination.

        Parameters
        ----------
        source : str
        destination : str

        """
        name = Path(source).name
        format = "zip"
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))

        if os.path.exists(os.path.join(destination, f"{name}.{format}")):
            lprint(
                "Previously existing model will be deleted before saving the new model"
            )
            os.remove(os.path.join(destination, f"{name}.{format}"))

        shutil.make_archive(name, format, archive_from, archive_to)

        try:
            shutil.move(f"{name}.{format}", destination)
        except shutil.Error as e:
            lprint(e)

    def save_model(self, model_path):
        """Saves the latest trained model next to the input image file.

        Parameters
        ----------
        model_path : str

        """
        # Save the model first
        self.it.save(model_path)

        # Make archive for the model
        self.archive_model(model_path, os.path.dirname(model_path))

        # clean the model folder
        self.clean_model_folder(model_path)
