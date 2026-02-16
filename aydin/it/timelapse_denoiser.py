"""Timelapse denoiser for large time-series image data.

This module provides `TimelapseDenoiser`, a specialized denoiser for
long timelapses that may not fit in memory. It uses temporal context
from neighboring frames and long-range averages to achieve temporally
consistent denoising.
"""

from typing import Tuple

import numpy

from aydin.io import imread
from aydin.io.io import mapped_tiff
from aydin.it.base import ImageTranslatorBase
from aydin.util.log.log import aprint, asection


class TimelapseDenoiser:
    """Timelapse denoiser for large time-series image data.

    Specialized for scaling up denoising to very long timelapses that may
    not fit in memory. Particularly useful for large 3D+t datasets.

    For each time point, features are collected from neighboring time points
    (fine temporal window) and long-range averages or medians (coarse temporal
    window). This helps achieve good temporal consistency.

    To protect against self-hallucinations and speed up denoising, a separate
    denoiser is trained per time point. It is advised to turn on spatial
    features. For timelapses with very large 3D stacks, it is recommended
    to use tiling.

    Attributes
    ----------
    translator : ImageTranslatorBase
        The underlying image translator used for per-timepoint denoising.
    fine_temporal_window : int
        Number of neighboring timepoints on each side used as fine-grained
        temporal features.
    coarse_temporal_window : int
        Number of exponentially-spaced long-range temporal windows on each
        side used as coarse temporal features.
    use_median : bool
        If True, use median for coarse temporal aggregation; otherwise use mean.
    """

    def __init__(
        self,
        translator: ImageTranslatorBase,
        fine_temporal_window: int = 1,
        coarse_temporal_window: int = 7,
        use_median=True,
    ):
        """Construct a timelapse denoiser.

        Parameters
        ----------
        translator : ImageTranslatorBase
            The underlying image translator to use for per-timepoint denoising.
        fine_temporal_window : int
            Number of neighboring timepoints on each side to use as
            fine-grained temporal features. Default is 1.
        coarse_temporal_window : int
            Number of exponentially-spaced long-range temporal windows
            on each side to use as coarse temporal features. Default is 7.
        use_median : bool
            If True, use median for coarse temporal aggregation.
            If False, use mean. Default is True.
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
        """Denoise an image given file paths for input and output.

        Reads the input image from disk, creates a memory-mapped output
        TIFF file, and runs the denoising pipeline.

        Parameters
        ----------
        input_image_path : str
            Path to the input image file.
        denoised_image_path : str
            Path where the denoised image will be saved as TIFF.
        batch_dims : tuple of bool, optional
            Specifies which axes are batch dimensions.
        channel_dims : tuple of bool, optional
            Specifies which axes are channel dimensions.
        tile_size : int, optional
            Suggested tile size for tiled inference.
        interval : tuple of (int, int), optional
            Time point range (start, end) to denoise. If None, all timepoints.
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
        """Denoise a timelapse image array one timepoint at a time.

        Ideally, the input array should not be fully loaded in memory
        but should be a lazy-loading array (e.g. memory-mapped TIFF).

        Parameters
        ----------
        input_image_array : numpy.ndarray
            Image array to denoise. First dimension is time.
        denoised_image_array : numpy.ndarray, optional
            Pre-allocated output array. If None, a new array is created.
        batch_dims : tuple of bool, optional
            Specifies which axes are batch dimensions.
        channel_dims : tuple of bool, optional
            Specifies which axes are channel dimensions.
        tile_size : int, optional
            Suggested tile size for tiled inference.
        interval : tuple of (int, int), optional
            Time point range (start, end) to denoise. If None, all timepoints.

        Returns
        -------
        numpy.ndarray
            The denoised image array.
        """
        with asection(
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
        """Denoise a single timepoint using temporal context features.

        Constructs input features from the fine and coarse temporal windows
        around the given timepoint, trains the translator, and translates.

        Parameters
        ----------
        tpi : int
            Index of the timepoint to denoise.
        num_time_points : int
            Total number of timepoints in the timelapse.
        input_image_array : numpy.ndarray
            Full timelapse input image array.
        denoised_image_array : numpy.ndarray
            Output array where the denoised timepoint is written.
        batch_dims : tuple of bool
            Batch dimension specification.
        channel_dims : tuple of bool
            Channel dimension specification.
        tile_size : int or None
            Tile size for tiled inference.
        """
        with asection(
            f"Denoising time point: {tpi} of shape: {input_image_array[tpi].shape}"
        ):
            ftw = self.fine_temporal_window
            with asection(
                f"Adding fine temporal feature channels for range: [-{ftw},{ftw}]"
            ):

                def get_relative_timpepoint(rel_index):
                    """Retrieve a timepoint frame relative to the current one.

                    Parameters
                    ----------
                    rel_index : int
                        Relative offset from the current timepoint index.

                    Returns
                    -------
                    numpy.ndarray
                        The image frame at the clamped index.
                    """
                    index = tpi + rel_index
                    index = min(num_time_points - 1, index)
                    index = max(0, index)
                    aprint(f'Fine features delta={rel_index} ')
                    return input_image_array[index]

                fine_window_list = [
                    get_relative_timpepoint(i) for i in range(-ftw, +ftw + 1)
                ]

            ctw = self.coarse_temporal_window
            with asection(
                f"Added coarse temporal feature channels for range: [-{ctw},{ctw}]"
            ):

                def get_average(extent):
                    """Compute temporal average or median over a range of frames.

                    For positive extents, averages frames from (tpi+1) to
                    (tpi+extent). For negative extents, averages frames from
                    (tpi+extent) to (tpi-1). Uses median or mean depending
                    on the ``use_median`` setting.

                    Parameters
                    ----------
                    extent : int
                        Signed distance from the current timepoint. Positive
                        for future frames, negative for past frames.

                    Returns
                    -------
                    numpy.ndarray
                        Aggregated (averaged or median) image frame, or
                        zeros if the range is empty.
                    """
                    if extent > 0:
                        index = tpi + extent
                        index = min(num_time_points - 1, index)

                        if index <= tpi:
                            return numpy.zeros_like(input_image_array[0])

                        aprint(f'Slice: {[tpi + 1, index + 1]}')
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
                        divisor = abs(tpi - (index))
                        if divisor == 0:
                            return numpy.zeros_like(input_image_array[0])
                        average_stack /= divisor

                    else:
                        index = tpi + extent
                        index = max(0, index)
                        if index >= tpi:
                            return numpy.zeros_like(input_image_array[0])

                        aprint(f'Slice: {[index, tpi - 1 + 1]}')
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
                        divisor = abs((tpi + 1) - index)
                        if divisor == 0:
                            return numpy.zeros_like(input_image_array[0])
                        average_stack /= divisor

                    aprint(f'Coarse temporal feature extent={extent}')
                    return average_stack

                coarse_window_list_past = [
                    get_average(ftw * 2**i) for i in range(1, ctw + 1)
                ]
                coarse_window_list_future = [
                    get_average(-ftw * 2**i) for i in range(1, ctw + 1)
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

            aprint(f'Channel dims: {channel_dims_tp} ')
            aprint(f'Force J-invariance tuple: {force_jinv} ')
            aprint(f'Pass-through channels: {self.translator._passthrough_channels} ')

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
