"""PyTorch CNN-based image translator.

This module provides `ImageTranslatorCNNTorch`, a convolutional neural network
based image translator using PyTorch. Supports pluggable model architectures
and training methods.
"""

import importlib
import inspect
import json
import pkgutil
from os.path import join
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import aydin.nn.models as nnmodels
from aydin.it.base import ImageTranslatorBase
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train
from aydin.nn.utils.center_smoothing import apply_center_smoothing
from aydin.util.log.log import aprint, asection


class ImageTranslatorCNNTorch(ImageTranslatorBase):
    """PyTorch CNN-based image translator.

    Provides a flexible framework for CNN-based image translation using PyTorch.
    Supports pluggable model architectures (e.g. JINet) and training methods
    (e.g. Noise2Self, Noise2Target).

    Attributes
    ----------
    model : torch.nn.Module or None
        The PyTorch model instance used for training and inference.
    model_class : type
        The PyTorch model class, either passed directly or resolved from
        a string name.
    model_kwargs : dict or None
        Additional keyword arguments passed to the model constructor.
    training_method : callable or None
        Training function (e.g. ``n2s_train`` or ``n2t_train``). If None,
        automatically selected based on whether training is self-supervised.
    training_method_kwargs : dict or None
        Additional keyword arguments passed to the training method.
    spacetime_ndim : int
        Number of spatio-temporal dimensions (2 or 3), set during training.
    """

    def __init__(
        self,
        model: Union[str, nn.Module] = "jinet",
        model_kwargs: Dict = None,
        training_method: Callable = None,
        training_method_kwargs: Dict = None,
        center_smoothing: bool = False,
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
        center_smoothing : bool
            If True, apply post-training center pixel smoothing to
            JINet-style models with DilatedConv layers.
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
        self.center_smoothing = center_smoothing
        self.stop_fitting = False
        self._stop_flag = None

    def __repr__(self):
        """Return a string representation of the PyTorch CNN translator."""
        return f"<{self.__class__.__name__}, model={self.model}, training_method={self.training_method}>"

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
        with asection(f"Saving 'CNN' image translator to {path}"):
            frozen = super().save(path)
            self.save_cnn(path)
        return frozen

    def save_cnn(self, path: str):
        """Save the PyTorch CNN model weights and metadata to disk.

        Saves model metadata (class name, module, spacetime_ndim,
        model_kwargs) as JSON and model weights via ``torch.save``.

        Parameters
        ----------
        path : str
            Directory path to save the model files to.
        """
        if self.model is not None:
            from aydin.nn.training_methods.n2s_shiftconv import ShiftConvWrapper

            is_shiftconv = isinstance(self.model, ShiftConvWrapper)
            base_model = self.model.base_model if is_shiftconv else self.model

            # Save model metadata as JSON
            metadata = {
                'class_name': self.model_class.__name__,
                'module_name': self.model_class.__module__,
                'spacetime_ndim': getattr(self, 'spacetime_ndim', None),
                'model_kwargs': self.model_kwargs,
                'shiftconv': is_shiftconv,
            }
            with open(join(path, 'torch_model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save base model state dict (not wrapper state dict)
            torch.save(base_model.state_dict(), join(path, 'torch_model_weights.pth'))
            aprint("PyTorch CNN model saved.")
        else:
            aprint("There is no model to save yet.")

    def __getstate__(self):
        """Customize pickle state to exclude non-serializable fields.

        Returns
        -------
        dict
            Object state with nn.Module and function references excluded.
        """
        state = super().__getstate__()
        # Remove non-JSON-serializable fields
        state.pop('model', None)
        state.pop('training_method', None)
        state.pop('_stop_flag', None)
        return state

    def _load_internals(self, path: str):
        """Load PyTorch model state from disk.

        Reads JSON metadata to reconstruct the model architecture, then
        loads saved weights via ``model.load_state_dict``.

        Parameters
        ----------
        path : str
            Directory path to load the model from.
        """
        with asection(f"Loading 'cnn' image translator from {path}"):
            # Load metadata
            with open(join(path, 'torch_model_metadata.json'), 'r') as f:
                metadata = json.load(f)

            # Reconstruct model class
            module = importlib.import_module(metadata['module_name'])
            model_class = getattr(module, metadata['class_name'])
            self.model_class = model_class

            # Build model kwargs
            model_args = {'spacetime_ndim': metadata['spacetime_ndim']}
            if metadata.get('model_kwargs'):
                model_args.update(metadata['model_kwargs'])

            # Instantiate model and load weights
            base_model = model_class(**model_args)
            base_model.load_state_dict(
                torch.load(join(path, 'torch_model_weights.pth'), weights_only=True)
            )

            # Wrap with ShiftConvWrapper if the model was trained with shiftconv
            if metadata.get('shiftconv'):
                from aydin.nn.training_methods.n2s_shiftconv import ShiftConvWrapper

                self.model = ShiftConvWrapper(
                    base_model, spacetime_ndim=metadata['spacetime_ndim']
                )
            else:
                self.model = base_model

            self.model.eval()
            aprint("PyTorch CNN model loaded.")

    def stop_training(self):
        """Request early stopping of the current training process.

        Sets the ``stop_fitting`` flag and the shared mutable stop flag
        dict that is passed to the training method.
        """
        self.stop_fitting = True
        if self._stop_flag is not None:
            self._stop_flag['stop'] = True

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

        # Find the model class: look for a class ending with "Model"
        # This handles naming mismatches like res_unet -> ResidualUNetModel
        class_name = [
            x for x in dir(response) if x.endswith('Model') and not x.startswith('_')
        ][0]

        model_class = getattr(response, class_name)

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
        # Reset stop flags
        self.stop_fitting = False
        self._stop_flag = {'stop': False}

        # Little heuristic to decide on training method if it is not specified
        if not self.training_method:
            self.training_method = (
                n2s_train if input_image is target_image else n2t_train
            )

        # If a model instance is not passed, create one
        if not self.model:
            self.model = self.model_class(**self._get_model_args(input_image))

        # Generate a dict of all arguments we can pass.
        # Include both singular/plural forms so the argument filter matches
        # whichever the training method expects (n2s uses input_image,
        # n2t uses input_images).
        training_method_args = {
            "input_image": input_image,
            "input_images": input_image,
            "target_image": target_image,
            "target_images": target_image,
            "model": self.model,
            "stop_fitting_flag": self._stop_flag,
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

        # For shiftconv training, wrap the model so inference uses the
        # same rotate-shift-unrotate pipeline the model was trained with.
        from aydin.nn.training_methods.n2s_shiftconv import (
            ShiftConvWrapper,
            n2s_shiftconv_train,
        )

        if self.training_method is n2s_shiftconv_train:
            self.model = ShiftConvWrapper(
                self.model, spacetime_ndim=self.spacetime_ndim
            )

        # Apply center smoothing after training if requested
        if self.center_smoothing and self.model is not None:
            apply_center_smoothing(
                self.model,
                spacetime_ndim=self.spacetime_ndim,
            )

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
            device = next(self.model.parameters()).device
            x = torch.as_tensor(input_image, dtype=torch.float32).to(device)

            # Determine required spatial alignment from model.
            # If model is a ShiftConvWrapper, look at the base_model.
            base = getattr(self.model, 'base_model', self.model)
            nb_unet_levels = getattr(base, 'nb_unet_levels', 3)
            divisor = 2**nb_unet_levels
            spatial_dims = x.shape[2:]  # skip B, C

            # Compute padding needed for each spatial dim
            pad_amounts = []
            for s in reversed(spatial_dims):
                remainder = s % divisor
                pad = (divisor - remainder) % divisor
                pad_amounts.extend([0, pad])  # (before, after) for this dim

            needs_padding = any(p > 0 for p in pad_amounts)
            if needs_padding:
                x = torch.nn.functional.pad(x, pad_amounts, mode='replicate')

            self.model.eval()
            with torch.no_grad():
                result = self.model(x)

            # Crop back to original spatial shape
            if needs_padding:
                slices = [slice(None), slice(None)]  # B, C
                for i, s in enumerate(spatial_dims):
                    slices.append(slice(0, s))
                result = result[tuple(slices)]

            return result.cpu().numpy()
        else:
            raise ValueError("A model is needed to infer on...")
