from typing import Optional, Union, List, Tuple

from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import lsection, lprint


class ImageTranslatorCNNTorch(ImageTranslatorBase):
    def __init__(
            self,
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
            jinv
    ):
        raise NotImplementedError()

    def _translate(
            self,
            input_image,
            image_slice=None,
            whole_image_shape=None
    ):
        raise NotImplementedError()
