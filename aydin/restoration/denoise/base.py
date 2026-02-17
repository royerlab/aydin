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
from aydin.util.log.log import aprint


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
        """Initialise the denoising restoration base.

        Parameters
        ----------
        variant : str, optional
            The specific algorithm variant to use (e.g. ``'butterworth'``,
            ``'cb'``). When ``None``, the default variant for the subclass
            is used.
        """
        self.variant = variant

        self.it = None

    def __repr__(self):
        """Return a string representation of the denoiser.

        Returns
        -------
        str
            A string showing the class name, variant, and image translator.
        """
        return f"<{self.__class__.__name__}, variant={self.variant}, self.it={self.it}>"

    @property
    @abstractmethod
    def configurable_arguments(self):
        """Return the configurable arguments exposed on the GUI and CLI.

        Returns
        -------
        dict
            A nested dictionary keyed by implementation name (e.g.
            ``'Classic-butterworth'``). Each value is a dictionary whose
            entries describe the parameter groups (e.g. ``'calibration'``,
            ``'it'``, ``'regressor'``) with their arguments, defaults,
            annotations, and reference class.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def implementations(self):
        """Return the list of discovered implementation variant names.

        Returns
        -------
        list of str
            Variant names prefixed with the method class name (e.g.
            ``'Classic-butterworth'``, ``'Noise2SelfFGR-cb'``).
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_running(self):
        """Stop the currently running training or inference process.

        Signals the underlying image translator to halt. Useful for
        cancelling long-running operations from the GUI.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, noisy_image, *, batch_axes=None, channel_axes=None, **kwargs):
        """Train the denoiser on a noisy image.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to train on.
        batch_axes : array_like, optional
            Indices of batch axes.
        channel_axes : array_like, optional
            Indices of channel axes.
        **kwargs
            Additional keyword arguments passed to the underlying translator.
        """
        raise NotImplementedError()

    @abstractmethod
    def denoise(self, noisy_image, *, batch_axes=None, channel_axes=None, **kwargs):
        """Denoise an image using the trained model.

        Parameters
        ----------
        noisy_image : numpy.ndarray
            The noisy input image to denoise.
        batch_axes : array_like, optional
            Indices of batch axes.
        channel_axes : array_like, optional
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
                f"No class matching"
                f" '{implementation_class_name}'"
                f" found in {module.name}"
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
        """Archive a model folder as a zip file in a destination directory.

        Parameters
        ----------
        source : str
            Path to the model folder to archive.
        destination : str
            Directory where the zip archive will be placed.
        """
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        name = Path(source).name
        archive_format = "zip"
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))

        if os.path.exists(os.path.join(destination, f"{name}.{archive_format}")):
            aprint(
                "Previously existing model will be deleted before saving the new model"
            )
            os.remove(os.path.join(destination, f"{name}.{archive_format}"))

        archive_base = os.path.join(destination, name)
        shutil.make_archive(archive_base, archive_format, archive_from, archive_to)

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

        aprint(f"Loading image translator from: {model_path}")
        shutil.unpack_archive(model_path, os.path.dirname(model_path), "zip")
        self.it = ImageTranslatorBase.load(model_path[:-4])
