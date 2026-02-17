"""Classical image denoiser implementation.

This module provides `ImageDenoiserClassic`, an image translator that uses
classical denoising methods (Butterworth, Gaussian, NLM, Total Variation, etc.)
with automatic parameter calibration via J-invariance.
"""

import importlib
from typing import List, Optional, Tuple, Union

import numpy

from aydin.it import classic_denoisers
from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import aprint, asection


class ImageDenoiserClassic(ImageTranslatorBase):
    """Classical image denoiser using traditional signal processing methods.

    Wraps classical denoising algorithms and automatically calibrates their
    parameters using J-invariance. Each channel is calibrated independently
    by selecting the batch with highest variance for parameter optimization.
    <notgui>
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
            from the image content. If 'center' is passed then only the default
            single center voxel blind-spot is used.

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

        self.method = method
        response = importlib.import_module(classic_denoisers.__name__ + '.' + method)

        self.calibration_function = response.__getattribute__(
            "calibrate_denoise_" + method
        )

        self.max_voxels_for_training = max_voxels_for_training

        self._memory_needed = 0

        self.main_channel = main_channel

        with asection("Classic image translator"):
            aprint(f"method: {method}")
            aprint(f"main channel: {main_channel}")

    def __repr__(self):
        """Return a string representation of the classic denoiser."""
        return (
            f"<{self.__class__.__name__}, "
            f"method={self.method}, "
            f"max_voxels_for_training="
            f"{self.max_voxels_for_training}>"
        )

    def save(self, path: str):
        """Save the classic denoiser model to disk.

        Parameters
        ----------
        path : str
            Directory path to save the model to.

        Returns
        -------
        str
            JSON string of the serialized model.
        """
        with asection(f"Saving 'classic' image denoiser to {path}"):
            frozen = super().save(path)

        return frozen

    def _load_internals(self, path: str):
        """Load classic denoiser internal state from disk.

        For the classic denoiser, no additional internal state needs
        to be loaded beyond JSON deserialization.

        Parameters
        ----------
        path : str
            Directory path to load from.
        """
        with asection(f"Loading 'classic' image denoiser from {path}"):
            pass

    def __getstate__(self):
        """Customize pickle state for serialization.

        Returns
        -------
        dict
            Object state dictionary.
        """
        state = self.__dict__.copy()
        return state

    def _train(
        self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        """Calibrate the classical denoiser on the input image.

        For each channel, selects the batch with highest standard deviation
        and calibrates denoising parameters using the selected method.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        target_image : numpy.ndarray
            Shape-normalized target image (unused for classical denoisers).
        train_valid_ratio : float
            Fraction of data for validation (unused for classical denoisers).
        callback_period : int
            Callback period in seconds (unused for classical denoisers).
        jinv : bool or tuple of bool, optional
            J-invariance flag (unused; classical methods use built-in calibration).
        """
        with asection(
            f"Training image translator from image of shape {input_image.shape}:"
        ):
            shape = input_image.shape
            num_channels = shape[1]

            self.best_parameters = []
            self.denoising_functions = []

            # We calibrate per channel
            for channel_index in range(num_channels):
                aprint(f'Calibrating denoiser on channel {channel_index}')
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
                aprint(f"Best parameters: {best_parameters}")

    def _estimate_memory_needed_and_available(self, image):
        """Estimate memory requirements for the classical denoiser.

        Parameters
        ----------
        image : numpy.ndarray
            The image to estimate memory requirements for.

        Returns
        -------
        tuple of (float, float)
            A tuple of (memory_needed, memory_available) in bytes.
        """
        _, available = super()._estimate_memory_needed_and_available(image)

        return self._memory_needed, available

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Denoise an input image using the calibrated classical method.

        Applies the per-channel denoising function with the best parameters
        found during calibration.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        image_slice : tuple of slice, optional
            Slice indicating where this tile sits within the whole image.
        whole_image_shape : tuple of int, optional
            Shape of the full image (before tiling).

        Returns
        -------
        numpy.ndarray
            Denoised image with same shape as input.
        """
        shape = input_image.shape
        num_batches = shape[0]
        num_channels = shape[1]

        denoised_image = numpy.empty_like(input_image)

        for batch_index in range(num_batches):
            for channel_index in range(num_channels):
                aprint(
                    f'Denoising image for batch: '
                    f'{batch_index} and channel: '
                    f'{channel_index}'
                )
                best_parameters = self.best_parameters[channel_index]
                denoising_function = self.denoising_functions[channel_index]
                image = input_image[batch_index, channel_index]
                denoised_image[batch_index, channel_index] = denoising_function(
                    image, **best_parameters
                )

        return denoised_image
