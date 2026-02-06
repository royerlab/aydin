"""Base class for all denoising restoration methods.

This module defines :class:`DenoiseRestorationBase`, the abstract base class
for Aydin's high-level denoising pipelines (Classic, Noise2Self FGR,
Noise2Self CNN). It provides common infrastructure for discovering
implementations, managing transforms, and saving/loading models.
"""

import importlib
import inspect
import os
import pkgutil
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import lprint


class DenoiseRestorationBase(ABC):
    """Abstract base class for denoising restoration methods.

    Subclasses implement specific denoising strategies (classical,
    self-supervised FGR, self-supervised CNN) while this base class
    provides implementation discovery, argument inspection, model
    archiving, and a common train/denoise interface.

    Attributes
    ----------
    variant : str or None
        The specific algorithm variant (e.g. ``'butterworth'``, ``'cb'``).
    it : ImageTranslatorBase or None
        The underlying image translator, set after training.
    """

    def __init__(self, variant: str = None):
        self.variant = variant

        self.it = None

    def __repr__(self):
        return f"<{self.__class__.__name__}, variant={self.variant}, self.it={self.it}>"

    @property
    @abstractmethod
    def configurable_arguments(self):
        """Returns the configurable arguments that will be exposed
        on GUI and CLI.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def implementations(self):
        """Returns the list of discovered implementations for given method."""
        raise NotImplementedError()

    @abstractmethod
    def stop_running(self):
        """Method to stop running restoration instance."""
        raise NotImplementedError()

    @abstractmethod
    def train(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Train the denoiser on a noisy image.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to train on.
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments passed to the underlying translator.
        """
        raise NotImplementedError()

    @abstractmethod
    def denoise(self, noisy_image, *, batch_axes=None, chan_axes=None, **kwargs):
        """Denoise an image using the trained model.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to denoise.
        batch_axes : array_like, optional
            Indices of batch axes.
        chan_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments passed to the underlying translator.

        Returns
        -------
        numpy.ndarray
            The denoised image.
        """
        raise NotImplementedError()

    @staticmethod
    def get_implementations_in_a_module(module):
        """Discover non-package, non-base submodules in the given package.

        Parameters
        ----------
        module : module
            Python package to inspect.

        Returns
        -------
        list
            List of :class:`pkgutil.ModuleInfo` objects for discovered
            implementations.
        """
        return [
            x
            for x in pkgutil.iter_modules(tuple(module.__path__))
            if not x.ispkg and x.name != 'base'
        ]

    @staticmethod
    def get_class_implementation_kwargs(package, module, implementation_class_name):
        """Extract constructor keyword arguments from a class implementation.

        Parameters
        ----------
        package : module
            Parent package containing the module.
        module : pkgutil.ModuleInfo
            Module containing the target class.
        implementation_class_name : str
            Name (case-insensitive substring) of the class to inspect.

        Returns
        -------
        dict
            Dictionary with keys ``'arguments'``, ``'defaults'``,
            ``'annotations'``, and ``'reference_class'``.

        Raises
        ------
        ValueError
            If no matching class is found in the module.
        """
        response = importlib.import_module(package.__name__ + '.' + module.name)
        matching = [
            x for x in dir(response) if implementation_class_name.lower() in x.lower()
        ]
        if not matching:
            raise ValueError(
                f"No class matching '{implementation_class_name}' found in {module.name}"
            )
        elem = matching[0]  # class name

        class_itself = response.__getattribute__(elem)
        fullargspec = inspect.getfullargspec(class_itself.__init__)

        defaults = fullargspec.defaults or ()
        args_offset = len(fullargspec.args) - len(defaults)

        args = {
            "arguments": fullargspec.args[args_offset:],
            "defaults": defaults,
            "annotations": fullargspec.annotations,
            "reference_class": class_itself,
        }

        return args

    @staticmethod
    def get_function_implementation_kwargs(
        package, module, implementation_function_name
    ):
        """Extract keyword arguments from a function implementation.

        Parameters
        ----------
        package : module
            Parent package containing the module.
        module : pkgutil.ModuleInfo
            Module containing the target function.
        implementation_function_name : str
            Exact name of the function to inspect.

        Returns
        -------
        dict
            Dictionary with keys ``'arguments'``, ``'defaults'``,
            ``'annotations'``, and ``'reference_class'`` (the function
            object itself).
        """
        response = importlib.import_module(package.__name__ + '.' + module.name)

        function_itself = response.__getattribute__(implementation_function_name)

        fullargspec = inspect.getfullargspec(function_itself)

        defaults = fullargspec.defaults or ()
        args_offset = len(fullargspec.args) - len(defaults)

        args = {
            "arguments": fullargspec.args[args_offset:],
            "defaults": defaults,
            "annotations": fullargspec.annotations,
            "reference_class": function_itself,
        }

        return args

    @staticmethod
    def clean_model_folder(model_folder):
        """Remove the model folder and all its contents.

        Parameters
        ----------
        model_folder : str
            Path to the model folder to delete.
        """
        shutil.rmtree(model_folder)

    @staticmethod
    def archive(source, destination):
        """Archive a model folder as a zip file and move it to a destination.

        Parameters
        ----------
        source : str
            Path to the model folder to archive.
        destination : str
            Directory where the zip archive will be placed.
        """
        name = Path(source).name
        archive_format = "zip"
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))

        if os.path.exists(os.path.join(destination, f"{name}.{archive_format}")):
            lprint(
                "Previously existing model will be deleted before saving the new model"
            )
            os.remove(os.path.join(destination, f"{name}.{archive_format}"))

        shutil.make_archive(name, archive_format, archive_from, archive_to)

        try:
            shutil.move(f"{name}.{archive_format}", destination)
        except shutil.Error as e:
            lprint(e)

    def save(self, model_path):
        """Save the trained model as a zip archive.

        Saves the image translator to ``model_path``, creates a zip archive,
        and removes the uncompressed folder.

        Parameters
        ----------
        model_path : str
            Directory path where the model will be saved before archiving.
        """
        # Save the model first
        self.it.save(model_path)

        # Make archive for the model
        self.archive(model_path, os.path.dirname(model_path))

        # clean the model folder
        self.clean_model_folder(model_path)

    def load(self, model_path: str):
        """Load a previously saved model from a zip archive.

        Unpacks the archive and restores the image translator into
        ``self.it``.

        Parameters
        ----------
        model_path : str
            Full path to the model zip file (e.g. ``'/path/to/model.zip'``).
        """

        lprint(f"Loading image translator from: {model_path}")
        shutil.unpack_archive(model_path, os.path.dirname(model_path), "zip")
        self.it = ImageTranslatorBase.load(model_path[:-4])
