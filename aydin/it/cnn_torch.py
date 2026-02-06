"""PyTorch CNN-based image translator.

This module provides `ImageTranslatorCNNTorch`, a convolutional neural network
based image translator using PyTorch. Supports pluggable model architectures
and training methods.
"""

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import aydin.nn.models as nnmodels
from aydin.it.base import ImageTranslatorBase
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train
from aydin.util.log.log import lprint, lsection


class ImageTranslatorCNNTorch(ImageTranslatorBase):
    """PyTorch CNN-based image translator.

    Provides a flexible framework for CNN-based image translation using PyTorch.
    Supports pluggable model architectures (e.g. JINet) and training methods
    (e.g. Noise2Self, Noise2Target).
    """

    def __init__(
        self,
        model: Union[str, nn.Module] = "jinet",
        model_kwargs: Dict = None,
        training_method: Callable = None,
        training_method_kwargs: Dict = None,
        blind_spots: Optional[Union[str, List[Tuple[int]]]] = None,
        tile_min_margin: int = 8,
        tile_max_margin: Optional[int] = None,
        max_memory_usage_ratio: float = 0.9,
        max_tiling_overhead: float = 0.1,
    ):
        """Construct a PyTorch CNN-based image translator.

        Parameters
        ----------
        model : str or nn.Module
            Either a model name string (e.g. 'jinet') that will be
            looked up from aydin.nn.models, or a pre-instantiated
            PyTorch nn.Module.
        model_kwargs : dict, optional
            Additional keyword arguments passed to the model constructor
            when model is specified as a string.
        training_method : callable, optional
            Training function to use. If None, automatically selects
            n2s_train (self-supervised) or n2t_train (supervised).
        training_method_kwargs : dict, optional
            Additional keyword arguments passed to the training method.
        blind_spots : str or list of tuple of int, optional
            Blind-spot specification. See `ImageTranslatorBase` for details.
        tile_min_margin : int
            Minimal width of tile margin in voxels.
        tile_max_margin : int, optional
            Maximal width of tile margin in voxels.
        max_memory_usage_ratio : float
            Maximum allowed memory load, value within [0, 1]. Default is 0.9.
        max_tiling_overhead : float
            Maximum allowed margin overhead during tiling. Default is 0.1.
        """
        super().__init__(
            blind_spots=blind_spots,
            tile_min_margin=tile_min_margin,
            tile_max_margin=tile_max_margin,
            max_memory_usage_ratio=max_memory_usage_ratio,
            max_tiling_overhead=max_tiling_overhead,
        )

        # Check if a model instance passed
        if isinstance(model, nn.Module):
            self.model, self.model_class = model, model.__class__
        else:
            self.model = None

            # If desired model name is passed as a string
            self.model_class = self._get_model_class_from_string(
                model if isinstance(model, str) else "jinet"
            )

        self.model_kwargs = model_kwargs
        self.training_method = training_method
        self.training_method_kwargs = training_method_kwargs

    def __repr__(self):
        """Return a string representation of the PyTorch CNN translator."""
        return f"<{self.__class__.__name__}, model={self.model}, training_method={self.training_method}"

    def save(self, path: str):
        """Save the PyTorch CNN translator model to disk.

        Parameters
        ----------
        path : str
            Directory path to save the model to.

        Returns
        -------
        str
            JSON string of the serialized model state.
        """
        with lsection(f"Saving 'CNN' image translator to {path}"):
            frozen = super().save(path)
            self.save_cnn(path)
        return frozen

    def save_cnn(self, path: str):
        """Save the PyTorch CNN model weights to disk.

        Parameters
        ----------
        path : str
            Directory path to save the model files to.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        if self.model is not None:
            # serialize model to JSON:
            raise NotImplementedError()
        else:
            lprint("There is no model to save yet.")

    def __getstate__(self):
        """Customize pickle state for serialization.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        # exclude fields below that should/cannot be saved properly:
        # del state['early_stopping']
        # del state['reduce_learning_rate']
        # del state['checkpoint']
        # del state['model']
        # del state['loss_history']
        # del state['infmodel']
        # del state['validation_images']
        raise NotImplementedError()

    def _load_internals(self, path: str):
        """Load PyTorch model state from disk.

        Parameters
        ----------
        path : str
            Directory path to load the model from.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        with lsection(f"Loading 'cnn' image translator from {path}"):
            # load JSON and create model:
            # self.model =
            raise NotImplementedError()

    def stop_training(self):
        """Stops currently running training within the instance by turning the flag
        true for early stop callback.

        """
        # self.stop_fitting = True
        raise NotImplementedError()

    @staticmethod
    def _get_model_class_from_string(model_name):
        """Look up a model class by name from the aydin.nn.models package.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g. 'jinet').

        Returns
        -------
        type
            The model class corresponding to the given name.
        """
        model_modules = [
            x
            for x in pkgutil.iter_modules(tuple(nnmodels.__path__))
            if not x.ispkg and x.name != 'base'
        ]
        module_of_interest = [
            module for module in model_modules if module.name == model_name
        ][0]

        response = importlib.import_module(
            "aydin.nn.models" + '.' + module_of_interest.name
        )

        class_name = [x for x in dir(response) if model_name + "model" in x.lower()][0]

        model_class = response.__getattribute__(class_name)

        return model_class

    @staticmethod
    def _get_function_args(function):
        """Returns name of arguments of a given function.

        Parameters
        ----------
        function : Callable
            The function of interest.

        Returns
        -------
        List[str]
            List of argument names for given function

        """
        return [param.name for param in inspect.signature(function).parameters.values()]

    def _get_model_args(self, input_image):
        """Build keyword arguments for model construction.

        Determines the number of spatio-temporal dimensions from the
        input image and merges with any user-provided model_kwargs.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image.

        Returns
        -------
        dict
            Keyword arguments for the model constructor.

        Raises
        ------
        ValueError
            If the number of spatio-temporal dimensions is not 2 or 3.
        """
        self.spacetime_ndim = input_image.ndim - 2
        if self.spacetime_ndim not in [2, 3]:
            raise ValueError("Number of spacetime dimensions have to be either 2 or 3.")

        args = {"spacetime_ndim": self.spacetime_ndim}
        if self.model_kwargs:
            args |= self.model_kwargs

        return args

    def _train(
        self,
        input_image,
        target_image,
        train_valid_ratio,  # TODO: should this parameter be here?
        callback_period,
        jinv,
    ):
        """Train the PyTorch CNN model on the given images.

        Automatically selects the training method (Noise2Self or Noise2Target)
        if not specified, constructs the model if needed, and runs training.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        target_image : numpy.ndarray
            Shape-normalized target image.
        train_valid_ratio : float
            Fraction of data for validation.
        callback_period : int
            Callback period in seconds.
        jinv : bool or tuple of bool, optional
            J-invariance flag.
        """
        # Little heuristic to decide on training method if it is not specified
        if not self.training_method:
            self.training_method = (
                n2s_train if input_image is target_image else n2t_train
            )

        # If a model instance is not passed, create one
        if not self.model:
            self.model = self.model_class(**self._get_model_args(input_image))

        # Generate a dict of all arguments we can pass
        training_method_args = {
            "input_image": input_image,
            "target_image": target_image,
            "model": self.model,
        }
        if self.training_method_kwargs:
            training_method_args |= self.training_method_kwargs

        # Filter the arguments for specific training_method
        filtered_training_method_args = {
            key: value
            for key, value in training_method_args.items()
            if key in self._get_function_args(self.training_method)
        }

        # Start training
        self.training_method(**filtered_training_method_args)

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Translate (denoise) an input image using the trained PyTorch CNN.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        image_slice : tuple of slice, optional
            Slice indicating tile position within the whole image.
        whole_image_shape : tuple of int, optional
            Shape of the full image before tiling.

        Returns
        -------
        numpy.ndarray
            Denoised output image.

        Raises
        ------
        ValueError
            If no trained model is available.
        """
        if self.model:
            return (
                self.model(
                    torch.Tensor(input_image).to(
                        next(self.model.parameters()).device
                    )  # Trick to get the model device
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            raise ValueError("A model is needed to infer on...")
