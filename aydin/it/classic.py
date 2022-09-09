import importlib
from typing import Optional, Union, List, Tuple
import numpy

from aydin.it import classic_denoisers
from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import lsection, lprint


class ImageDenoiserClassic(ImageTranslatorBase):
    """
    Classic Image Denoiser
    """

    def __init__(
        self,
        method: str = "butterworth",
        main_channel: Optional[int] = None,
        max_voxels_for_training: Optional[int] = None,
        calibration_kwargs: Optional[dict] = None,
        blind_spots: Optional[Union[str, List[Tuple[int]]]] = None,
        tile_min_margin: int = 8,
        tile_max_margin: Optional[int] = None,
        max_memory_usage_ratio: float = 0.9,
        max_tiling_overhead: float = 0.1,
    ):
        """Constructs a Classic image denoiser.

        Parameters
        ----------
        method: str
            Name of classical denoising method.

        main_channel: optional int
            By default the denoiser is calibrated per channel.
            To speed up denoising you can pick one channel index
            to use during calibration and used to denoise all channels.

        max_voxels_for_training : int, optional
            Maximum number of the voxels that can be
            used for training.

        calibration_kwargs : Optional[dict]
            Depending on the classic denoising algorithm you can use this parameter
            to pass the calibration parameters. (advanced) (hidden)

        blind_spots : Optional[Union[str,List[Tuple[int]]]]
            List of voxel coordinates (relative to receptive field center) to
            be included in the blind-spot. For example, you can enter:
            '<axis>#<radius>' to extend the blindspot along a given axis by a
            certain radius. For example, for an image of dimension 3, 'x#1'
            extends the blind spot to cover voxels of relative coordinates:
            (0,0,0),(0,1,0), and (0,-1,0). If you want to extend both in x and y,
            enter: 'x#1,y#1' by comma separating between axis. To specify the
            axis you can use integer indices, or 'x', 'y', 'z', and 't'
            (dimension order is tzyx with x being always the last dimension).
            If None is passed then the blindspots are automatically discovered
            from the image content. If 'center' is passed then no additional
            blindspots to the center pixel are considered.  If 'center' is passed
            then only the default single center voxel blind-spot is used.

        tile_min_margin : int
            Minimal width of tile margin in voxels.
            (advanced)

        tile_max_margin : Optional[int]
            Maximal width of tile margin in voxels.
            (advanced)

        max_memory_usage_ratio : float
            Maximum allowed memory load, value must be within [0, 1]. Default is 90%.
            (advanced)

        max_tiling_overhead : float
            Maximum allowed margin overhead during tiling. Default is 10%.
            (advanced)
        """
        super().__init__(
            blind_spots=blind_spots,
            tile_min_margin=tile_min_margin,
            tile_max_margin=tile_max_margin,
            max_memory_usage_ratio=max_memory_usage_ratio,
            max_tiling_overhead=max_tiling_overhead,
        )

        self.calibration_kwargs = (
            {} if calibration_kwargs is None else calibration_kwargs
        )

        response = importlib.import_module(classic_denoisers.__name__ + '.' + method)

        self.calibration_function = response.__getattribute__(
            "calibrate_denoise_" + method
        )

        self.max_voxels_for_training = max_voxels_for_training

        self._memory_needed = 0

        self.main_channel = main_channel

        with lsection("Classic image translator"):
            lprint(f"method: {method}")
            lprint(f"main channel: {main_channel}")

    def save(self, path: str):
        """Saves a 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------
        frozen

        """
        with lsection(f"Saving 'classic' image denoiser to {path}"):
            frozen = super().save(path)

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image denoiser from {path}"):
            pass

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def _train(
        self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        with lsection(
            f"Training image translator from image of shape {input_image.shape}:"
        ):
            shape = input_image.shape
            num_channels = shape[1]

            self.best_parameters = []
            self.denoising_functions = []

            # We calibrate per channel
            for channel_index in range(num_channels):
                lprint(f'Calibrating denoiser on channel {channel_index}')
                channel_image = input_image[:, channel_index]

                # for a given channel we find the best batch to use:
                # We pick the batch with highest variance:
                variance_list = [numpy.std(i) for i in channel_image]
                batch_index = variance_list.index(max(variance_list))

                # We pick that batch image:
                image = channel_image[batch_index]

                (
                    denoising_function,
                    best_parameters,
                    memory_requirements,
                ) = self.calibration_function(
                    image, blind_spots=self.blind_spots, **self.calibration_kwargs
                )

                # Add obtained best parameters to the list per channel:
                self.denoising_functions.append(denoising_function)
                self.best_parameters.append(best_parameters)
                self._memory_needed = memory_requirements

    def _estimate_memory_needed_and_available(self, image):
        """

        Parameters
        ----------
        image

        Returns
        -------

        """
        _, available = super()._estimate_memory_needed_and_available(image)

        return self._memory_needed, available

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Internal method that translates an input image on the basis of the trained model.

        Parameters
        ----------
        input_image
            input image
        image_slice
        whole_image_shape

        Returns
        -------
        numpy.ArrayLike
            translated image

        """
        shape = input_image.shape
        num_batches = shape[0]
        num_channels = shape[1]

        denoised_image = numpy.empty_like(input_image)

        for batch_index in range(num_batches):
            for channel_index in range(num_channels):
                lprint(
                    f'Denoising image for batch: {batch_index} and channel: {channel_index}'
                )
                best_parameters = self.best_parameters[channel_index]
                denoising_function = self.denoising_functions[channel_index]
                image = input_image[batch_index, channel_index]
                denoised_image[batch_index, channel_index] = denoising_function(
                    image, **best_parameters
                )

        return denoised_image
