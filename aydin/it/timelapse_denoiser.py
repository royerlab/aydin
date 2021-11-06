from typing import Tuple

import numpy

from aydin.io import imread
from aydin.io.io import mapped_tiff
from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import lprint, lsection


class TimelapseDenoiser:
    """TimeLapse denoiser
    This translator is specialised for scaling up denoising to very long timelapses that can't fit in memory.
    This makes particularly sense for very large 3D+t datasets.
    For each time point we collect features from time points before and after as well as long-range averages (or medians)
    This helps to get good temporal consistency. To protect against self-hallucinations and speed up denoising, we train
    one denoiser per time point. It is advised to turn on spatial features. For timelapses with very large 3D stacks, it is
    recomended to use tiling.

    Attributes
    ----------

    Notes
    -----

    """

    def __init__(
        self,
        translator: ImageTranslatorBase,
        fine_temporal_window: int = 1,
        coarse_temporal_window: int = 7,
        use_median=True,
    ):
        """Constructs a timelapse denoiser given an image translator and the following parameters:

        Parameters
        ----------
        translator : ImageTranslatorBase
        coarse_temporal_window : int
        fine_temporal_window : int
        use_median : bool

        """
        self.use_median = use_median
        self.coarse_temporal_window = coarse_temporal_window
        self.fine_temporal_window = fine_temporal_window
        self.translator = translator

    def denoise_paths(
        self,
        input_image_path: str,
        denoised_image_path: str,
        batch_dims: Tuple[int] = None,
        channel_dims: Tuple[int] = None,
        tile_size=None,
        interval=None,
    ):
        """Convenience method to denoise an image from a path

        Parameters
        ----------
        input_image_path : str
        denoised_image_path : str
        batch_dims : tuple
        channel_dims : tuple
        tile_size
        interval

        """
        input_image_array, _ = imread(input_image_path)

        with mapped_tiff(
            denoised_image_path, input_image_array.shape, input_image_array.dtype
        ) as denoised_image_array:
            self.denoise(
                input_image_array,
                denoised_image_array,
                batch_dims,
                channel_dims,
                tile_size=tile_size,
                interval=interval,
            )

    def denoise(
        self,
        input_image_array: numpy.ndarray,
        denoised_image_array: numpy.ndarray = None,
        batch_dims: Tuple[int] = None,
        channel_dims: Tuple[int] = None,
        tile_size=None,
        interval=None,
    ):
        """Denoises image.
        Ideally, the array should not be fully loaded in memory but should be a 'lazy-loading' array.

        Parameters
        ----------
        input_image_array : numpy.ndarray
            image to denoise
        denoised_image_array : numpy.ndarray
            denoised image
        batch_dims : tuple
            tuple specifying which are the batch dimensions
        channel_dims : tuple
            tuple specifying which are the channel dimensions
        """
        with lsection(
            f"Denoising image with dimensions {input_image_array.shape} along first dimension"
        ):
            num_time_points = input_image_array.shape[0]

            # set default batch_dim and channel_dim values:
            if batch_dims is None:
                batch_dims = (False,) * len(input_image_array.shape)
            if channel_dims is None:
                channel_dims = (False,) * len(input_image_array.shape)

            if denoised_image_array is None:
                denoised_image_array = numpy.empty_like(input_image_array)

            tp_begin = 0 if interval is None else interval[0]
            tp_end = num_time_points if interval is None else interval[1]
            for tpi in range(tp_begin, tp_end):
                self._denoise_single_timepoint(
                    tpi,
                    num_time_points,
                    input_image_array,
                    denoised_image_array,
                    batch_dims,
                    channel_dims,
                    tile_size,
                )

            return denoised_image_array

    def _denoise_single_timepoint(
        self,
        tpi,
        num_time_points,
        input_image_array,
        denoised_image_array,
        batch_dims,
        channel_dims,
        tile_size,
    ):
        """Private method to denoise single given timepoint.

        Parameters
        ----------
        tpi
        num_time_points
        input_image_array
        denoised_image_array
        batch_dims
        channel_dims
        tile_size

        """
        with lsection(
            f"Denoising time point: {tpi} of shape: {input_image_array[tpi].shape}"
        ):
            ftw = self.fine_temporal_window
            with lsection(
                f"Adding fine temporal feature channels for range: [-{ftw},{ftw}]"
            ):

                def get_relative_timpepoint(rel_index):
                    index = tpi + rel_index
                    index = min(num_time_points - 1, index)
                    index = max(0, index)
                    lprint(f'Fine features delta={rel_index} ')
                    return input_image_array[index]

                fine_window_list = [
                    get_relative_timpepoint(i) for i in range(-ftw, +ftw + 1)
                ]

            ctw = self.coarse_temporal_window
            with lsection(
                f"Added coarse temporal feature channels for range: [-{ctw},{ctw}]"
            ):

                def get_average(extent):

                    if extent > 0:
                        index = tpi + extent
                        index = min(num_time_points - 1, index)

                        if index <= tpi:
                            return numpy.zeros_like(input_image_array[0])

                        lprint(f'Slice: {[tpi + 1, index + 1]}')
                        average_stack = input_image_array[tpi + 1 : index + 1]
                        average_stack = average_stack.astype(numpy.float32)
                        if self.use_median:
                            average_stack = numpy.median(
                                average_stack, axis=0, keepdims=False
                            )
                        else:
                            average_stack = numpy.sum(
                                average_stack, axis=0, keepdims=False
                            )
                        average_stack /= abs(tpi - (index))

                    else:
                        index = tpi + extent
                        index = max(0, index)
                        if index >= tpi:
                            return numpy.zeros_like(input_image_array[0])

                        lprint(f'Slice: {[index, tpi - 1 + 1]}')
                        average_stack = input_image_array[index : tpi - 1 + 1]
                        average_stack = average_stack.astype(numpy.float32)
                        if self.use_median:
                            average_stack = numpy.median(
                                average_stack, axis=0, keepdims=False
                            )
                        else:
                            average_stack = numpy.sum(
                                average_stack, axis=0, keepdims=False
                            )
                        average_stack /= abs((tpi + 1) - index)

                    lprint(f'Coarse temporal feature extent={extent}')
                    return average_stack

                coarse_window_list_past = [
                    get_average(ftw * 2 ** i) for i in range(1, ctw + 1)
                ]
                coarse_window_list_future = [
                    get_average(-ftw * 2 ** i) for i in range(1, ctw + 1)
                ]

            input_array_for_tp = numpy.stack(
                fine_window_list + coarse_window_list_past + coarse_window_list_future
            )
            target_array_for_tp = input_image_array[tpi : tpi + 1]

            channel_dims_tp = (True,) + tuple(c for c in channel_dims[1:])
            force_jinv = (
                (False,) * ftw + (True,) + (False,) * ftw + (False,) * (2 * ctw)
            )
            self.translator._passthrough_channels = (False,) * (2 * ftw + 1) + (
                True,
            ) * (2 * ctw)

            lprint(f'Channel dims: {channel_dims_tp} ')
            lprint(f'Force J-invariance tuple: {force_jinv} ')
            lprint(f'Pass-through channels: {self.translator._passthrough_channels} ')

            self.translator.train(
                input_array_for_tp,
                target_array_for_tp,
                batch_axes=batch_dims,
                channel_axes=channel_dims_tp,
                jinv=force_jinv,
            )

            self.translator.translate(
                input_array_for_tp,
                denoised_image_array[tpi],
                batch_axes=batch_dims,
                channel_axes=channel_dims_tp,
                tile_size=tile_size,
            )
