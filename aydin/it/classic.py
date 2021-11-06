import importlib
from typing import Optional
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
        calibration_kwargs=None,
        **kwargs,
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

        kwargs : dict
            Keyword arguments.
        """
        super().__init__(**kwargs)

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
            # TODO: implement!

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image denoiser from {path}"):
            # TODO: implement!
            pass
            # self.best_parameters = FeatureGeneratorBase.load(path)

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # TODO: implement!
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
                ) = self.calibration_function(image, **self.calibration_kwargs)

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

        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
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
