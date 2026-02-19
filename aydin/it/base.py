"""Base classes for the Image Translator framework.

This module defines the abstract base class `ImageTranslatorBase` that provides
the core interface and shared functionality for all image translation implementations,
including training, inference, tiling, transform management, and serialization.
"""

import gc
import math
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Optional, Tuple, Union

import jsonpickle
import numpy
from numpy import array2string

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError
from aydin.it.normalisers.shape import ShapeNormaliser
from aydin.it.transforms.base import ImageTransformBase
from aydin.util.array.nd import nd_split_slices, remove_margin_slice
from aydin.util.log.log import aprint, asection
from aydin.util.misc.json import encode_indent
from aydin.util.offcore.offcore import offcore_array

# Maximum number of voxels per tile for tiled processing
_MAX_VOXELS_PER_TILE = 768**3


class ImageTranslatorBase(ABC):
    """Abstract base class for image translators.

    Provides the core interface for training and translating (denoising) images.
    Handles batch/channel axis normalization, tiled inference for large images,
    image transforms (preprocessing/postprocessing), blind-spot specification,
    and model serialization.

    Subclasses must implement `_train`, `_translate`, and `_load_internals`.

    Attributes
    ----------
    self_supervised : bool
        Whether the translator is operating in self-supervised mode
        (input and target are the same image).
    monitor : object or None
        Monitor object for tracking training progress and callbacks.
    blind_spots : str, list of tuple of int, or None
        Blind-spot specification for self-supervised training. Controls
        which voxels relative to the center are excluded during training.
    tile_max_margin : int or None
        Maximum margin width in voxels for tiled processing.
    tile_min_margin : int
        Minimum margin width in voxels for tiled processing.
    transforms_list : list of ImageTransformBase
        Ordered list of image transforms applied as preprocessing (forward)
        and postprocessing (reverse).
    max_memory_usage_ratio : float
        Maximum fraction of available memory to use (0 to 1).
    max_tiling_overhead : float
        Maximum allowed margin overhead fraction during tiling.
    max_voxels_per_tile : int
        Maximum number of voxels per tile for tiled processing.
    callback_period : int
        Minimum period in seconds between training progress callbacks.
    last_callback_time_sec : float
        Timestamp of the last callback invocation.
    loss_history : object or None
        Training loss history after training completes.
    """

    def __init__(
        self,
        monitor=None,
        blind_spots: Optional[Union[str, List[Tuple[int]]]] = None,
        tile_min_margin: int = 8,
        tile_max_margin: Optional[int] = None,
        max_memory_usage_ratio: float = 0.9,
        max_tiling_overhead: float = 0.1,
    ):
        """Construct an ImageTranslatorBase.

        Parameters
        ----------
        monitor : object, optional
            Monitor object for tracking training progress and callbacks.

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
            Maximum allowed memory load, value must be within [0, 1]. Default
            is 90%. (advanced)

        max_tiling_overhead : float
            Maximum allowed margin overhead during tiling. Default is 10%.
            (advanced)

        """
        self.self_supervised = False
        self.monitor = monitor
        self.blind_spots: Optional[Union[str, List[Tuple[int]]]] = blind_spots
        self.tile_max_margin = tile_max_margin
        self.tile_min_margin = tile_min_margin

        self.transforms_list: List[ImageTransformBase] = []

        self.max_memory_usage_ratio = max_memory_usage_ratio
        self.max_tiling_overhead = max_tiling_overhead
        self.max_voxels_per_tile = _MAX_VOXELS_PER_TILE

        self.callback_period = 3
        self.last_callback_time_sec = -math.inf

        self.loss_history = None

    @property
    def max_spacetime_ndim(self) -> Optional[int]:
        """Maximum supported number of spatio-temporal dimensions.

        Subclasses may override this to impose a limit (e.g. CNN translators
        support only 2D and 3D).  Return ``None`` for no limit.
        """
        return None

    def add_transform(self, transform: ImageTransformBase, sort: bool = True):
        """Add a transform to the preprocessing/postprocessing pipeline.

        Parameters
        ----------
        transform : ImageTransformBase
            The image transform to add.
        sort : bool
            If True, re-sort transforms by priority after adding.
        """
        self.transforms_list.append(transform)
        if sort:
            self.transforms_list = sorted(
                self.transforms_list, key=lambda t: t.priority
            )

    def clear_transforms(self):
        """Remove all transforms from the preprocessing/postprocessing pipeline."""
        self.transforms_list.clear()

    def transform_preprocess_image(self, image):
        """Apply all transforms in order as preprocessing steps.

        Parameters
        ----------
        image : numpy.ndarray
            Input image to preprocess.

        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        with asection("transform preprocess:"):
            for transform in self.transforms_list:
                aprint(f"applying transform: {transform}")
                try:
                    image = transform.preprocess(image)
                except Exception as e:
                    error_message = str(e).replace('\n', ', ')
                    aprint(
                        f"Preprocessing failed for {transform} with: {error_message} "
                    )
                    import sys
                    import traceback

                    traceback.print_exception(*sys.exc_info())

        return image

    def transform_postprocess_image(self, image):
        """Apply all transforms in reverse order as postprocessing steps.

        Parameters
        ----------
        image : numpy.ndarray
            Image to postprocess.

        Returns
        -------
        numpy.ndarray
            Postprocessed image.
        """
        with asection("transform postprocess"):
            for transform in reversed(self.transforms_list):
                aprint(f"applying transform: {transform}")
                try:
                    image = transform.postprocess(image)
                except Exception as e:
                    error_message = str(e).replace('\n', ', ')
                    aprint(
                        f"Postprocessing failed for {transform} with: {error_message}"
                    )
                    import sys
                    import traceback

                    traceback.print_exception(*sys.exc_info())

        return image

    @abstractmethod
    def _train(
        self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        """Train the model on shape-normalized input and target images.

        Subclasses must implement this method to define their training logic.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        target_image : numpy.ndarray
            Shape-normalized target image with same shape as input_image.
        train_valid_ratio : float
            Fraction of data to use for validation (e.g. 0.1 for 10%).
        callback_period : int
            Period in seconds between progress callbacks.
        jinv : bool or tuple of bool, optional
            Controls J-invariance (blind-spot) behavior during training.
        """
        raise NotImplementedError()

    def stop_training(self):
        """Request early stopping of an ongoing training process.

        The default implementation is a no-op. Subclasses should override
        this to support interrupting training.
        """
        pass

    @abstractmethod
    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Translate an input image tile into a denoised output image.

        Subclasses must implement this method to define their inference logic.

        Parameters
        ----------
        input_image : numpy.ndarray
            Shape-normalized input image with shape (B, C, *spatial_dims).
        image_slice : tuple of slice, optional
            Slice tuple indicating where this tile sits within the whole image.
        whole_image_shape : tuple of int, optional
            Shape of the full image (before tiling).

        Returns
        -------
        numpy.ndarray
            Translated (denoised) output image with same shape as input.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_internals(self, path: str):
        """Load model-specific internal state from disk.

        Subclasses must implement this to restore any state not captured
        by JSON serialization (e.g., neural network weights).

        Parameters
        ----------
        path : str
            Directory path to load from.
        """
        raise NotImplementedError()

    def _estimate_memory_needed_and_available(self, image):
        """Estimate memory requirements and availability for translating an image.

        Parameters
        ----------
        image : numpy.ndarray
            The image to estimate memory requirements for.

        Returns
        -------
        tuple of (float, float)
            A tuple of (memory_needed, memory_available) in bytes.
        """
        # TODO: make sure if this makes sense with GPU too
        # By default there is no memory needed and all memory is available
        try:
            import psutil

            return 0, psutil.virtual_memory().total
        except ImportError:
            aprint(
                "Warning: psutil is not installed — assuming 16 GB total memory. "
                "Install it with: pip install psutil"
            )
            return 0, 16e9

    def train(
        self,
        input_image,
        target_image=None,
        batch_axes=None,
        channel_axes=None,
        train_valid_ratio=0.1,
        callback_period=3,
        jinv=None,
    ):
        """Train the translator to map input images to target images.

        Handles batch/channel axis parsing, shape normalization,
        blind-spot auto-detection, transform preprocessing, and
        delegates to the subclass-specific `_train` method.

        Parameters
        ----------
        input_image : numpy.ndarray
            Input image array of arbitrary dimensionality.
        target_image : numpy.ndarray, optional
            Target image array. If None, uses input_image (self-supervised).
        batch_axes : tuple of bool or list of int, optional
            Specifies which axes are batch dimensions.
        channel_axes : tuple of bool or list of int, optional
            Specifies which axes are channel dimensions.
        train_valid_ratio : float
            Fraction of data reserved for validation. Default is 0.1.
        callback_period : int
            Period in seconds between monitoring callbacks. Default is 3.
        jinv : bool or tuple of bool, optional
            Controls J-invariance behavior. None means auto-detect.

        Raises
        ------
        ArrayShapeDoesNotMatchError
            If batch_axes length does not match image dimensions, or if
            input and target spatial dimensions do not match.
        """

        if target_image is None:
            target_image = input_image

        with asection(
            f"Learning to translate from image of dimensions "
            f"{str(input_image.shape)} to {str(target_image.shape)}, "
            f"batch_axes={batch_axes}, channel_axes={channel_axes}, "
            f"jinv={jinv}."
        ):

            aprint('Running garbage collector...')
            gc.collect()

            # If we use the same image for input and output
            # then we are in a self-supervised setting:
            self.self_supervised = input_image is target_image

            if self.self_supervised:
                aprint('Training is self-supervised.')
            else:
                aprint('Training is supervised.')

            # Let's apply the transforms:
            input_image = self.transform_preprocess_image(input_image)
            target_image = (
                input_image
                if self.self_supervised
                else self.transform_preprocess_image(target_image)
            )

            if batch_axes is None:  # set default batch_axes value:
                batch_axes = (False,) * len(input_image.shape)
            if channel_axes is None:
                channel_axes = (False,) * len(input_image.shape)

            # parse batch and channel dim args
            batch_axes, channel_axes = self.parse_axes_args(
                batch_axes, channel_axes, len(input_image.shape)
            )

            if len(batch_axes) != len(input_image.shape):
                raise ArrayShapeDoesNotMatchError(
                    'The length of batch_dims and input_image dimensions are different.'
                )

            # Shape normalisers to use:
            shape_normaliser = ShapeNormaliser(
                batch_axes=batch_axes, channel_axes=channel_axes
            )
            self.shape_normaliser = shape_normaliser

            # Axis normalisation:
            shape_normalised_input_image = shape_normaliser.normalise(input_image)
            shape_normalised_target_image = (
                shape_normalised_input_image
                if self.self_supervised
                else shape_normaliser.normalise(target_image)
            )
            self.target_shape_normaliser = shape_normaliser

            # Validate spacetime dimensions against translator limits:
            num_spacetime = shape_normalised_input_image.ndim - 2
            if (
                self.max_spacetime_ndim is not None
                and num_spacetime > self.max_spacetime_ndim
            ):
                spacetime_shape = shape_normalised_input_image.shape[2:]
                raise ValueError(
                    f"This translator supports at most "
                    f"{self.max_spacetime_ndim}D spatial data, but the image "
                    f"has {num_spacetime} spacetime dimensions "
                    f"{spacetime_shape} after shape normalization "
                    f"(image shape: {input_image.shape}, "
                    f"batch_axes={list(batch_axes)}, "
                    f"channel_axes={list(channel_axes)}). "
                    f"Mark one or more leading dimensions as batch axes "
                    f"(e.g. in the Dimensions tab) so that at most "
                    f"{self.max_spacetime_ndim} spatial dimensions remain."
                )

            # Automatic blind-spot discovery:
            if self.blind_spots is None:
                aprint(
                    "Automatic discovery of noise autocorrelation "
                    "and specification of N2S blind-spots activated!"
                )
                self.blind_spots, autocorrelogram = auto_detect_blindspots(
                    shape_normalised_input_image[0, 0]
                )
                aprint(f"Blind spots: {self.blind_spots}")
                autocorrelogram_values = numpy.unique(autocorrelogram)
                autocorrelogram_values = numpy.sort(autocorrelogram_values)[::-1][:5]
                auto_str = array2string(
                    autocorrelogram_values,
                    precision=4,
                    separator=',',
                    suppress_small=True,
                    threshold=128,
                    edgeitems=16,
                    sign='+',
                )
                aprint(f"Autocorrelogram unique values in decreasing order: {auto_str}")
            elif isinstance(self.blind_spots, str):
                # Number of spatio-temporal dims:
                st_ndim = shape_normalised_input_image.ndim - 2
                # Parse:
                self.blind_spots = self._parse_blind_spot_shorthand_notation(
                    self.blind_spots, st_ndim
                )

            # Verify that input and target images have same shape:
            # We do this after normalisation because that's easier
            # we only compare the spatio-temporal dimensions ( not batches or channels)
            if (
                shape_normalised_input_image.shape[2:]
                != shape_normalised_target_image.shape[2:]
            ):
                raise ArrayShapeDoesNotMatchError(
                    'Input and Output image shape does not match!'
                )

            self._train(
                shape_normalised_input_image,
                shape_normalised_target_image,
                train_valid_ratio=train_valid_ratio,
                callback_period=callback_period,
                jinv=jinv,
            )

    def translate(
        self,
        input_image,
        translated_image=None,
        batch_axes=None,
        channel_axes=None,
        tile_size=None,
    ):
        """Translate (denoise) an input image using the trained model.

        Handles tiled inference for large images, shape normalization,
        and transform postprocessing.

        Parameters
        ----------
        input_image : numpy.ndarray
            Input image to translate.
        translated_image : numpy.ndarray, optional
            Pre-allocated output array. If None, a new array is created.
        batch_axes : tuple of bool or list of int, optional
            Specifies which axes are batch dimensions.
        channel_axes : tuple of bool or list of int, optional
            Specifies which axes are channel dimensions.
        tile_size : int, optional
            Suggested tile size for tiled inference. Use 0 to disable tiling.
            If None, tiling is determined automatically.

        Returns
        -------
        numpy.ndarray
            The translated (denoised) image.

        Raises
        ------
        ArrayShapeDoesNotMatchError
            If batch_axes length does not match image dimensions.
        """

        with asection(
            f"Predicting output image from input image of "
            f"dimension {input_image.shape}, "
            f"batch_axes={batch_axes}, "
            f"channel_axes={channel_axes}"
        ):

            # Let's apply the transforms:
            input_image = self.transform_preprocess_image(input_image)

            # set default batch_axes and channel_axes values:
            if batch_axes is None:
                batch_axes = (False,) * len(input_image.shape)
            if channel_axes is None:
                channel_axes = (False,) * len(input_image.shape)

            # parse batch and chan dim args
            batch_axes, channel_axes = self.parse_axes_args(
                batch_axes, channel_axes, len(input_image.shape)
            )

            if len(batch_axes) != len(
                input_image.shape
            ):  # Sanity check when using not-default batch dims:
                raise ArrayShapeDoesNotMatchError(
                    "batch_dims does not have same number of "
                    "dimensions with input_image!"
                )

            # Number of spatio-temporal dimensions:
            num_spatiotemp_dim = sum(
                0 if b or c else 1 for b, c in zip(batch_axes, channel_axes)
            )

            # Shape normalisers to use:
            shape_normaliser = ShapeNormaliser(
                batch_axes=batch_axes, channel_axes=channel_axes
            )

            # First we normalise the input values:
            shape_normalised_input_image = shape_normaliser.normalise(input_image)

            # Validate spacetime dimensions against translator limits:
            num_spacetime = shape_normalised_input_image.ndim - 2
            if (
                self.max_spacetime_ndim is not None
                and num_spacetime > self.max_spacetime_ndim
            ):
                spacetime_shape = shape_normalised_input_image.shape[2:]
                raise ValueError(
                    f"This translator supports at most "
                    f"{self.max_spacetime_ndim}D spatial data, but the image "
                    f"has {num_spacetime} spacetime dimensions "
                    f"{spacetime_shape} after shape normalization "
                    f"(image shape: {input_image.shape}, "
                    f"batch_axes={list(batch_axes)}, "
                    f"channel_axes={list(channel_axes)}). "
                    f"Mark one or more leading dimensions as batch axes "
                    f"(e.g. in the Dimensions tab) so that at most "
                    f"{self.max_spacetime_ndim} spatial dimensions remain."
                )

            # Spatio-temporal shape:
            spatiotemp_shape = shape_normalised_input_image.shape[-num_spatiotemp_dim:]

            shape_normalised_translated_image = None

            if tile_size == 0:
                # we _force_ no tilling, this is _not_ the default.

                # We translate:
                shape_normalised_translated_image = self._translate(
                    shape_normalised_input_image,
                    whole_image_shape=shape_normalised_input_image.shape,
                )

            else:

                # We do need to do tiled inference because of a lack of memory
                # or because a small batch size was requested:

                normalised_input_shape = shape_normalised_input_image.shape

                # We get the tilling strategy:
                # tile_size, shape, min_margin, max_margin
                tilling_strategy, margins = self._get_tilling_strategy_and_margins(
                    shape_normalised_input_image,
                    self.max_voxels_per_tile,
                    self.tile_min_margin,
                    self.tile_max_margin,
                    suggested_tile_size=tile_size,
                )
                aprint(f"Tilling strategy: {tilling_strategy}")
                aprint(f"Margins for tiles: {margins} .")

                # tile slice objects (with and without margins):
                tile_slices_margins = list(
                    nd_split_slices(
                        normalised_input_shape, tilling_strategy, margins=margins
                    )
                )
                tile_slices = list(
                    nd_split_slices(normalised_input_shape, tilling_strategy)
                )

                # Number of tiles:
                number_of_tiles = len(tile_slices)
                aprint(f"Number of tiles (slices): {number_of_tiles}")

                # We create slice list:
                slicezip = zip(tile_slices_margins, tile_slices)

                counter = 1
                for slice_margin_tuple, slice_tuple in slicezip:
                    with asection(
                        f"Current tile: {counter}/{number_of_tiles}"
                        f", slice: {slice_tuple} "
                    ):

                        # We first extract the tile image:
                        input_image_tile = shape_normalised_input_image[
                            slice_margin_tuple
                        ].copy()

                        # We do the actual translation:
                        aprint("Translating...")
                        translated_image_tile = self._translate(
                            input_image_tile,
                            image_slice=slice_margin_tuple,
                            whole_image_shape=shape_normalised_input_image.shape,
                        )

                        # We compute the slice needed to cut out the margins:
                        aprint("Removing margins...")
                        remove_margin_slice_tuple = remove_margin_slice(
                            normalised_input_shape, slice_margin_tuple, slice_tuple
                        )

                        # We allocate -just in time- the translated
                        # array if needed: if the array is already
                        # provided, it must have the right dimensions.
                        if shape_normalised_translated_image is None:
                            translated_image_shape = (
                                shape_normalised_input_image.shape[:2]
                                + spatiotemp_shape
                            )
                            shape_normalised_translated_image = offcore_array(
                                shape=translated_image_shape,
                                dtype=translated_image_tile.dtype,
                                max_memory_usage_ratio=self.max_memory_usage_ratio,
                            )

                        # We plug in the batch without margins
                        # into the destination image:
                        aprint("Inserting translated batch into result image...")
                        shape_normalised_translated_image[slice_tuple] = (
                            translated_image_tile[remove_margin_slice_tuple]
                        )

                        counter += 1

            # Then we shape denormalise:
            shape_denormalised_translated_image = shape_normaliser.denormalise(
                shape_normalised_translated_image
            )

            # Let's undo the transforms:
            shape_denormalised_translated_image = self.transform_postprocess_image(
                shape_denormalised_translated_image
            )

            if translated_image is None:
                translated_image = shape_denormalised_translated_image
            else:
                translated_image[...] = shape_denormalised_translated_image

            return translated_image

    def _get_tilling_strategy_and_margins(
        self,
        image,
        max_voxels_per_tile,
        min_margin,
        max_margin,
        suggested_tile_size=None,
    ):
        """Determine the optimal tiling strategy and margins for inference.

        Computes how to split an image into tiles based on memory constraints,
        maximum voxels per tile, and suggested tile sizes.

        Parameters
        ----------
        image : numpy.ndarray
            The image to compute the tiling strategy for.
        max_voxels_per_tile : int
            Maximum number of voxels allowed per tile.
        min_margin : int
            Minimum margin width in voxels around each tile.
        max_margin : int or None
            Maximum margin width in voxels around each tile.
        suggested_tile_size : int, optional
            Suggested tile size. If None, determined automatically.

        Returns
        -------
        tuple of (tuple of int, tuple of int)
            A tuple of (tiling_strategy, margins) where tiling_strategy
            specifies the number of splits per dimension and margins
            specifies the overlap per dimension.
        """

        # We will store the batch strategy as a list of integers
        # representing the number of chunks per dimension:
        with asection("Determine tilling strategy:"):

            suggested_tile_size = (
                math.inf if suggested_tile_size is None else suggested_tile_size
            )

            # image shape:
            shape = image.shape
            num_spatio_temp_dim = num_spatiotemp_dim = len(shape) - 2

            aprint(f"image shape             = {shape}")
            aprint(f"max_voxels_per_tile     = {max_voxels_per_tile}")

            # Estimated amount of memory needed for storing all features:
            (
                estimated_memory_needed,
                total_memory_available,
            ) = self._estimate_memory_needed_and_available(image)
            aprint(f"Estimated amount of memory needed: {estimated_memory_needed}")

            # Available physical memory :
            total_memory_available *= self.max_memory_usage_ratio

            aprint(
                f"Available memory (we reserve 10% for 'comfort'): "
                f"{total_memory_available}"
            )

            # How much do we need to tile because of memory, if at all?
            split_factor_mem = estimated_memory_needed / total_memory_available
            aprint(
                f"How much do we need to tile because of memory? "
                f": {split_factor_mem} times."
            )

            # how much do we have to tile because of the limit
            # on the number of voxels per tile?
            split_factor_max_voxels = image.size / max_voxels_per_tile
            aprint(
                "How much do we need to tile because of the "
                "limit on the number of voxels per tile? "
                f": {split_factor_max_voxels} times."
            )

            # how much do we have to tile because of the
            # suggested tile size?
            split_factor_suggested_tile_size = image.size / (
                suggested_tile_size**num_spatio_temp_dim
            )
            aprint(
                "How much do we need to tile because of the "
                "suggested tile size? "
                f": {split_factor_suggested_tile_size} times."
            )

            # we keep the max:
            desired_split_factor = max(
                split_factor_mem,
                split_factor_max_voxels,
                split_factor_suggested_tile_size,
            )
            # We cannot split less than 1 time:
            desired_split_factor = max(1, int(math.ceil(desired_split_factor)))
            aprint(f"Desired split factor: {desired_split_factor}")

            # Number of batches:
            num_batches = shape[0]

            # Does the number of batches split the data enough?
            if num_batches < desired_split_factor:
                # Not enough splitting happening along the batch
                # dimension, we need to split further. How much?
                rest_split_factor = desired_split_factor / num_batches
                aprint(
                    "Not enough splitting happening along the "
                    "batch dimension, we need to split "
                    f"spatio-temp dims by: {rest_split_factor}"
                )

                # let's split the dimensions in a way proportional to their lengths:
                split_per_dim = (rest_split_factor / numpy.prod(shape[2:])) ** (
                    1 / num_spatio_temp_dim
                )
                aprint(f"Splitting per dimension: {split_per_dim}")

                # We split proportionally to each axis but do not
                # exceed the rest_split_factor per axis:
                spatiotemp_tilling_strategy = tuple(
                    max(
                        1,
                        min(
                            rest_split_factor,
                            int(math.ceil(split_per_dim * s)),
                        ),
                    )
                    for s in shape[2:]
                )

                tilling_strategy = (num_batches, 1) + spatiotemp_tilling_strategy
                aprint(f"Preliminary tilling strategy is: {tilling_strategy}")

                # We correct for eventual oversplitting by favouring
                # splitting over the front dimensions:
                current_splitting_factor = 1
                corrected_tilling_strategy = []
                split_factor_reached = False
                for i, s in enumerate(tilling_strategy):

                    if split_factor_reached:
                        corrected_tilling_strategy.append(1)
                    else:
                        corrected_tilling_strategy.append(s)
                        current_splitting_factor *= s

                    if current_splitting_factor >= desired_split_factor:
                        split_factor_reached = True

                tilling_strategy = tuple(corrected_tilling_strategy)

            else:
                tilling_strategy = (desired_split_factor, 1) + tuple(
                    1 for s in shape[2:]
                )

            aprint(f"Tilling strategy is: {tilling_strategy}")

            # Handles defaults:
            if max_margin is None:
                max_margin = math.inf
            if min_margin is None:
                min_margin = 0

            # First we estimate the shape of a tile:

            estimated_tile_shape = tuple(
                int(round(s / ts)) for s, ts in zip(shape[2:], tilling_strategy[2:])
            )
            aprint(f"The estimated tile shape is: {estimated_tile_shape}")

            # Limit margins:
            # We automatically set the margin of the tile size:
            # the max-margin factor guarantees that tilling will
            # incur no more than a given max tiling overhead:
            margin_factor = 0.5 * (
                ((1 + self.max_tiling_overhead) ** (1 / num_spatiotemp_dim)) - 1
            )
            margins = tuple(int(s * margin_factor) for s in estimated_tile_shape)

            # Limit the margin to something reasonable
            # (provided or automatically computed):
            margins = tuple(min(max_margin, m) for m in margins)
            margins = tuple(max(min_margin, m) for m in margins)

            # We add the batch and channel dimensions:
            margins = (0, 0) + margins

            # We only need margins if we split a dimension:
            margins = tuple(
                (0 if split == 1 else margin)
                for margin, split in zip(margins, tilling_strategy)
            )

            return tilling_strategy, margins

    def save(self, path: str):
        """Save the image translator model to disk.

        Serializes the translator state as JSON and writes it to
        the specified directory.

        Parameters
        ----------
        path : str
            Directory path to save the model to.

        Returns
        -------
        str
            JSON string of the serialized model.
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        aprint(f"Saving image translator to: {path}")
        with open(join(path, "image_translation.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """Load a previously saved image translator model from disk.

        Parameters
        ----------
        path : str
            Directory path to load the model from.

        Returns
        -------
        ImageTranslatorBase
            The restored image translator instance.
        """
        aprint(f"Loading image translator from: {path}")
        with open(join(path, "image_translation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed._load_internals(path)

        return thawed

    def __getstate__(self):
        """Customize pickle state to exclude non-serializable normaliser fields.

        Returns
        -------
        dict
            Object state dictionary with normalisers excluded.
        """
        state = self.__dict__.copy()
        if 'input_normaliser' in state:
            del state['input_normaliser']
        if 'target_normaliser' in state:
            del state['target_normaliser']
        state.pop('shape_normaliser', None)
        state.pop('target_shape_normaliser', None)
        return state

    @staticmethod
    def parse_axes_args(
        batch_axes: Union[List[int], List[bool]],
        channel_axes: Union[List[int], List[bool]],
        ndim: int,
    ):
        """Parse and validate batch and channel axis specifications.

        Accepts either boolean arrays or integer index arrays and converts
        them into a consistent boolean array representation.

        Parameters
        ----------
        batch_axes : list of int or list of bool
            Batch axis specification as boolean flags or integer indices.
        channel_axes : list of int or list of bool
            Channel axis specification as boolean flags or integer indices.
        ndim : int
            Total number of dimensions in the image.

        Returns
        -------
        tuple of (list of bool, list of bool)
            Validated boolean arrays for batch and channel axes.

        Raises
        ------
        Exception
            If axes indices are out of range, axes types are mixed,
            the number of spacetime axes is invalid (not 1-4),
            or any axis is marked as both batch and channel.
        """
        if (
            batch_axes == []
            or batch_axes is None
            or all(isinstance(x, bool) for x in batch_axes)
        ) and (
            channel_axes == []
            or channel_axes is None
            or all(isinstance(x, bool) for x in channel_axes)
        ):
            # check if it is all boolean values then check
            # if it is correct size then return
            batch_result, chan_result = batch_axes, channel_axes
        elif (
            batch_axes == []
            or batch_axes is None
            or all(isinstance(x, int) for x in batch_axes)
        ) and (
            channel_axes == []
            or channel_axes is None
            or all(isinstance(x, int) for x in channel_axes)
        ):
            # check if it is list of indices and can make
            # meaningful boolean arrays, if so return
            if any(i < 0 or i >= ndim for i in batch_axes) or any(
                i < 0 or i >= ndim for i in channel_axes
            ):
                raise Exception(
                    "No axes index can be smaller than zero or bigger than ndim-1!"
                )

            batch_result = [True if i in batch_axes else False for i in range(ndim)]
            chan_result = [True if i in channel_axes else False for i in range(ndim)]
        else:
            raise Exception(
                "Axes arguments has to be boolean arrays or integer index arrays!"
            )

        ndim_spacetime = ndim - (batch_result.count(True) + chan_result.count(True))
        if ndim_spacetime > 4 or ndim_spacetime < 1:
            aprint(batch_result)
            aprint(chan_result)
            aprint(
                ndim, batch_result.count(True), chan_result.count(True), ndim_spacetime
            )
            raise Exception(
                "Number of spacetime axes cannot be more "
                "than 4 and cannot be less than 1!"
            )

        if any(
            [batch_result[i] is True and chan_result[i] is True for i in range(ndim)]
        ):
            raise Exception("No axes can be both batch and channel axes!")

        return batch_result, chan_result

    @staticmethod
    def _parse_blind_spot_shorthand_notation(blind_spots: str, st_ndim: int):
        """Parse blind-spot shorthand notation into a list of coordinate tuples.

        Converts shorthand strings like 'x#1,y#2' into explicit lists of
        relative voxel coordinates for the blind-spot mask.

        Parameters
        ----------
        blind_spots : str
            Shorthand notation string. Use '<axis>#<radius>' format
            (e.g. 'x#1' or 'x#1,y#2'). Use 'center' for only the
            center voxel.
        st_ndim : int
            Number of spatio-temporal dimensions.

        Returns
        -------
        list of tuple of int
            List of relative voxel coordinate tuples for the blind-spot.
        """
        aprint(f"Blindspot shorthand notation detected: {blind_spots} ")
        # Replace commas with spaces:
        blind_spots = blind_spots.replace(',', ' ')
        # First split by white space:
        parts = blind_spots.split()

        # We accumulate parsed blind spots here:
        blind_spots_set = set()
        # To avoid confusiuon we always include the center pixel:
        blind_spots_set.add((0,) * st_ndim)

        if 'center' in blind_spots:
            # We don't extend!
            pass
        else:
            for part in parts:
                splitted_part = part.split('#')
                axis = splitted_part[0].strip()

                # Parse shorthand axis notation:
                if axis == 'x':
                    axis = st_ndim - 1 - 0
                elif axis == 'y':
                    axis = st_ndim - 1 - 1
                elif axis == 'z':
                    axis = st_ndim - 1 - 2
                elif axis == 't':
                    axis = st_ndim - 1 - 3
                else:
                    axis = int(axis)

                # Ensure axis is in range:
                axis = max(0, min(st_ndim - 1, axis))

                radius = int(splitted_part[1].strip())

                for r in range(-radius, radius + 1):
                    spot = (0,) * axis + (r,) + (0,) * (st_ndim - 1 - axis)
                    blind_spots_set.add(spot)

        blind_spots = list(blind_spots_set)

        aprint(f"Parsed blindspot from shorthand notation: {blind_spots} ")

        return blind_spots
