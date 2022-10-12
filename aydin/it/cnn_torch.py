import importlib
import inspect
import pkgutil

from torch import nn
from typing import Optional, Union, List, Tuple, Callable

from aydin.it.base import ImageTranslatorBase
import aydin.nn.models as nnmodels
from aydin.nn.training_methods.n2s import n2s_train
from aydin.nn.training_methods.n2t import n2t_train
from aydin.util.log.log import lsection, lprint


class ImageTranslatorCNNTorch(ImageTranslatorBase):
    def __init__(
        self,
        model: Union[str, nn.Module] = "jinet",
        training_method: Callable = None,
        patch_size: int = None,
        nb_epochs: int = 256,
        lr: float = 0.01,
        patience: int = 4,
        blind_spots: Optional[Union[str, List[Tuple[int]]]] = None,
        tile_min_margin: int = 8,
        tile_max_margin: Optional[int] = None,
        max_memory_usage_ratio: float = 0.9,
        max_tiling_overhead: float = 0.1,
    ):
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

        self.training_method = training_method
        self.patch_size = patch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.patience = patience

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------

        """
        with lsection(f"Saving 'CNN' image translator to {path}"):
            frozen = super().save(path)
            self.save_cnn(path)
        return frozen

    def save_cnn(self, path: str):
        if self.model is not None:
            # serialize model to JSON:
            raise NotImplementedError()
        else:
            lprint("There is no model to save yet.")

    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude fields below that should/cannot be saved properly:
        # del state['early_stopping']
        # del state['reduce_learning_rate']
        # del state['checkpoint']
        # del state['model']
        # del state['loss_history']
        # del state['infmodel']
        # del state['validation_images']
        raise NotImplementedError()

        return state

    def _load_internals(self, path: str):
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
        model_modules = [
            x
            for x in pkgutil.iter_modules(tuple(nnmodels.__path__))
            if not x.ispkg and x.name != 'base'
        ]
        module_of_interest = [
            module for module in model_modules if module.name == model_name
        ][0]

        response = importlib.import_module("aydin.nn.models"+'.'+module_of_interest.name)

        class_name = [x for x in dir(response) if model_name+"model" in x.lower()][0]

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

    def _train(
        self,
        input_image,
        target_image,
        train_valid_ratio,  # TODO: should this parameter be here?
        callback_period,
        jinv,
    ):
        # Little heuristic to decide on training method if it is not specified
        if not self.training_method:
            self.training_method = (
                n2s_train if input_image is target_image else n2t_train
            )

        # If a model instance is not passed, create one
        if not self.model:
            self.model = self.model_class(model_args)

        # Generate a dict of all arguments we can pass
        training_method_args = {
            "input_image": input_image,
            "target_image": target_image,
            "model": self.model,
            "nb_epochs": self.nb_epochs,
            "lr": self.lr,
            "patience": self.patience,
        }

        # Filter the arguments for specific training_method
        filtered_training_method_args = {
            key: value
            for key, value in training_method_args.items()
            if key in self._get_function_args(self.training_method)
        }

        # Start training
        self.training_method(**filtered_training_method_args)

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        raise NotImplementedError()
