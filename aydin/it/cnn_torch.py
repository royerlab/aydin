from torch import nn
from typing import Optional, Union, List, Tuple, Callable

from aydin.it.base import ImageTranslatorBase
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

        self.model = None
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

    def _train(
        self,
        input_image,
        target_image,
        train_valid_ratio,  # TODO: should this parameter be here?
        callback_period,
        jinv,
    ):
        if not self.training_method:
            self.training_method = n2s_train if input_image == target_image else n2t_train

        training_method_args = {
            "input_image": input_image,
            "target_image": target_image,
            "model": self.model,
            "nb_epochs": self.nb_epochs,
            "lr": self.lr,
            "patience": self.patience,
        }

        self.training_method(**training_method_args)

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        raise NotImplementedError()