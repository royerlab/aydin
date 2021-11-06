import gc
import math
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Union, List, Optional, Tuple
import jsonpickle
import numpy
import psutil

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError
from aydin.it.normalisers.shape import ShapeNormaliser
from aydin.it.transforms.base import ImageTransformBase
from aydin.util.array.nd import nd_split_slices, remove_margin_slice
from aydin.util.misc.json import encode_indent
from aydin.util.log.log import lprint, lsection
from aydin.util.offcore.offcore import offcore_array


class ImageTranslatorBase(ABC):
    """Image Translator base class

    Notes
    -----

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
        """Image Translator base class

        Parameters
        ----------
        monitor
        blind_spots : Optional[Union[str, List[Tuple[int]]]]
        tile_min_margin : int
        tile_max_margin : Optional[int]
        max_memory_usage_ratio : float
            Maximum allowed memory load.
        max_tiling_overhead : float
            Maximum allowed margin overhead during tiling.

        """
        self.self_supervised = False
        self.monitor = monitor
        self.blind_spots = blind_spots
        self.tile_max_margin = tile_max_margin
        self.tile_min_margin = tile_min_margin

        self.transforms_list: List[ImageTransformBase] = []

        self.max_memory_usage_ratio = max_memory_usage_ratio
        self.max_tiling_overhead = max_tiling_overhead
        self.max_voxels_per_tile = 768 ** 3

        self.callback_period = 3
        self.last_callback_time_sec = -math.inf

        self.loss_history = None

    def add_transform(self, transform: ImageTransformBase, sort: bool = True):
        """Adds the given transform to the self.transforms_list

        Parameters
        ----------
        transform : ImageTransformBase

        """
        self.transforms_list.append(transform)
        if sort:
            self.transforms_list = sorted(
                self.transforms_list, key=lambda t: t.priority
            )

    def clear_transforms(self):
        """Clears the transforms list"""
        self.transforms_list.clear()

    def _transform_preprocess_image(self, image):
        with lsection("transform preprocess:"):
            for transform in self.transforms_list:
                lprint(f"applying transform: {transform}")
                try:
                    image = transform.preprocess(image)
                except BaseException as e:
                    error_message = str(e).replace('\n', ', ')
                    lprint(
                        f"Preprocessing failed for {transform} with: {error_message} "
                    )
                    import traceback
                    import sys

                    traceback.print_exception(*sys.exc_info())

        return image

    def _transform_postprocess_image(self, image):
        with lsection("transform postprocess"):
            for transform in reversed(self.transforms_list):
                lprint(f"applying transform: {transform}")
                try:
                    image = transform.postprocess(image)
                except BaseException as e:
                    error_message = str(e).replace('\n', ', ')
                    lprint(
                        f"Postprocessing failed for {transform} with: {error_message}"
                    )
                    import traceback
                    import sys

                    traceback.print_exception(*sys.exc_info())

        return image

    @abstractmethod
    def _train(
        self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        """This function supposed to take normalized input image only

        Parameters
        ----------
        input_image
        target_image
        train_valid_ratio
        callback_period
        jinv

        Returns
        -------

        """
        raise NotImplementedError()

    def stop_training(self):
        pass

    @abstractmethod
    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Translates an input image into an output image according to the learned function

        Parameters
        ----------
        input_image
            input image
        image_slice
        whole_image_shape

        Returns
        -------

        """
        raise NotImplementedError()

    @abstractmethod
    def _load_internals(self, path: str):
        raise NotImplementedError()

    def _estimate_memory_needed_and_available(self, image):
        """

        Parameters
        ----------
        image

        Returns
        -------

        """
        # TODO: make sure if this makes sense with GPU too
        # By default there is no memory needed and all memory is available
        return 0, psutil.virtual_memory().total

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
        """Train to translate a given input image to a given output image.
        This has a lot of the machinery for batching and more...
        """

        if target_image is None:
            target_image = input_image

        with lsection(
            f"Learning to translate from image of dimensions {str(input_image.shape)} to {str(target_image.shape)}, batch_axes={batch_axes}, channel_axes={channel_axes}, jinv={jinv}."
        ):

            lprint('Running garbage collector...')
            gc.collect()

            # If we use the same image for input and output then we are in a self-supervised setting:
            self.self_supervised = input_image is target_image

            if self.self_supervised:
                lprint('Training is self-supervised.')
            else:
                lprint('Training is supervised.')

            # Let's apply the transforms:
            input_image = self._transform_preprocess_image(input_image)
            target_image = (
                input_image
                if self.self_supervised
                else self._transform_preprocess_image(target_image)
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

            # Axis normalisation:
            shape_normalised_input_image = shape_normaliser.normalise(input_image)
            shape_normalised_target_image = (
                shape_normalised_input_image
                if self.self_supervised
                else shape_normaliser.normalise(target_image)
            )

            # Automatic blind-spot discovery:
            if self.blind_spots == 'auto':
                lprint(
                    "Automatic discovery of noise autocorrelation and specification of N2S blind-spots activated!"
                )
                self.blind_spots = auto_detect_blindspots(
                    shape_normalised_input_image[0, 0]
                )[0]
                lprint(f"Blind spots: {self.blind_spots}")

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

        with lsection(
            f"Predicting output image from input image of dimension {input_image.shape}, batch_axes={batch_axes}, channel_axes={channel_axes}"
        ):

            # Let's apply the transforms:
            input_image = self._transform_preprocess_image(input_image)

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
                    'batch_dims does not have same number of dimensions with input_image!'
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
                lprint(f"Tilling strategy: {tilling_strategy}")
                lprint(f"Margins for tiles: {margins} .")

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
                lprint(f"Number of tiles (slices): {number_of_tiles}")

                # We create slice list:
                slicezip = zip(tile_slices_margins, tile_slices)

                counter = 1
                for slice_margin_tuple, slice_tuple in slicezip:
                    with lsection(
                        f"Current tile: {counter}/{number_of_tiles}, slice: {slice_tuple} "
                    ):

                        # We first extract the tile image:
                        input_image_tile = shape_normalised_input_image[
                            slice_margin_tuple
                        ].copy()

                        # We do the actual translation:
                        lprint("Translating...")
                        translated_image_tile = self._translate(
                            input_image_tile,
                            image_slice=slice_margin_tuple,
                            whole_image_shape=shape_normalised_input_image.shape,
                        )

                        # We compute the slice needed to cut out the margins:
                        lprint("Removing margins...")
                        remove_margin_slice_tuple = remove_margin_slice(
                            normalised_input_shape, slice_margin_tuple, slice_tuple
                        )

                        # We allocate -just in time- the translated array if needed:
                        # if the array is already provided, it must of course have the right dimensions...
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

                        # We plug in the batch without margins into the destination image:
                        lprint("Inserting translated batch into result image...")
                        shape_normalised_translated_image[
                            slice_tuple
                        ] = translated_image_tile[remove_margin_slice_tuple]

                        counter += 1

            # Then we shape denormalise:
            shape_denormalised_translated_image = shape_normaliser.denormalise(
                shape_normalised_translated_image
            )

            # Let's undo the transforms:
            shape_denormalised_translated_image = self._transform_postprocess_image(
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

        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:
        with lsection("Determine tilling strategy:"):

            suggested_tile_size = (
                math.inf if suggested_tile_size is None else suggested_tile_size
            )

            # image shape:
            shape = image.shape
            num_spatio_temp_dim = num_spatiotemp_dim = len(shape) - 2

            lprint(f"image shape             = {shape}")
            lprint(f"max_voxels_per_tile     = {max_voxels_per_tile}")

            # Estimated amount of memory needed for storing all features:
            (
                estimated_memory_needed,
                total_memory_available,
            ) = self._estimate_memory_needed_and_available(image)
            lprint(f"Estimated amount of memory needed: {estimated_memory_needed}")

            # Available physical memory :
            total_memory_available *= self.max_memory_usage_ratio

            lprint(
                f"Available memory (we reserve 10% for 'comfort'): {total_memory_available}"
            )

            # How much do we need to tile because of memory, if at all?
            split_factor_mem = estimated_memory_needed / total_memory_available
            lprint(
                f"How much do we need to tile because of memory? : {split_factor_mem} times."
            )

            # how much do we have to tile because of the limit on the number of voxels per tile?
            split_factor_max_voxels = image.size / max_voxels_per_tile
            lprint(
                f"How much do we need to tile because of the limit on the number of voxels per tile? : {split_factor_max_voxels} times."
            )

            # how much do we have to tile because of the suggested tile size?
            split_factor_suggested_tile_size = image.size / (
                suggested_tile_size ** num_spatio_temp_dim
            )
            lprint(
                f"How much do we need to tile because of the suggested tile size? : {split_factor_suggested_tile_size} times."
            )

            # we keep the max:
            desired_split_factor = max(
                split_factor_mem,
                split_factor_max_voxels,
                split_factor_suggested_tile_size,
            )
            # We cannot split less than 1 time:
            desired_split_factor = max(1, int(math.ceil(desired_split_factor)))
            lprint(f"Desired split factor: {desired_split_factor}")

            # Number of batches:
            num_batches = shape[0]

            # Does the number of batches split the data enough?
            if num_batches < desired_split_factor:
                # Not enough splitting happening along the batch dimension, we need to split further:
                # how much?
                rest_split_factor = desired_split_factor / num_batches
                lprint(
                    f"Not enough splitting happening along the batch dimension, we need to split spatio-temp dims by: {rest_split_factor}"
                )

                # let's split the dimensions in a way proportional to their lengths:
                split_per_dim = (rest_split_factor / numpy.prod(shape[2:])) ** (
                    1 / num_spatio_temp_dim
                )
                lprint(f"Splitting per dimension: {split_per_dim}")

                # We split proportionally to each axis but do not exceed the rest_split_factor per axis:
                spatiotemp_tilling_strategy = tuple(
                    max(1, min(rest_split_factor, int(math.ceil(split_per_dim * s))))
                    for s in shape[2:]
                )

                tilling_strategy = (num_batches, 1) + spatiotemp_tilling_strategy
                lprint(f"Preliminary tilling strategy is: {tilling_strategy}")

                # We correct for eventual oversplitting by favouring splitting over the front dimensions:
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

            lprint(f"Tilling strategy is: {tilling_strategy}")

            # Handles defaults:
            if max_margin is None:
                max_margin = math.inf
            if min_margin is None:
                min_margin = 0

            # First we estimate the shape of a tile:

            estimated_tile_shape = tuple(
                int(round(s / ts)) for s, ts in zip(shape[2:], tilling_strategy[2:])
            )
            lprint(f"The estimated tile shape is: {estimated_tile_shape}")

            # Limit margins:
            # We automatically set the margin of the tile size:
            # the max-margin factor guarantees that tilling will incur no more than a given max tiling overhead:
            margin_factor = 0.5 * (
                ((1 + self.max_tiling_overhead) ** (1 / num_spatiotemp_dim)) - 1
            )
            margins = tuple(int(s * margin_factor) for s in estimated_tile_shape)

            # Limit the margin to something reasonable (provided or automatically computed):
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
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------

        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving image translator to: {path}")
        with open(join(path, "image_translation.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """
        Returns an 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to load from.

        Returns
        -------

        """
        lprint(f"Loading image translator from: {path}")
        with open(join(path, "image_translation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed._load_internals(path)

        return thawed

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['input_normaliser']
        del state['target_normaliser']
        return state

    @staticmethod
    def parse_axes_args(
        batch_axes: Union[List[int], List[bool]],
        chan_axes: Union[List[int], List[bool]],
        ndim: int,
    ):
        """


        Parameters
        ----------
        batch_axes : Union[List[int], List[bool]]
        chan_axes : Union[List[int], List[bool]]
        ndim : int

        Returns
        -------

        """
        if (
            batch_axes == []
            or batch_axes is None
            or all(isinstance(x, bool) for x in batch_axes)
        ) and (
            chan_axes == []
            or chan_axes is None
            or all(isinstance(x, bool) for x in chan_axes)
        ):
            # check if it is all boolean values then check if it is correct size then return
            batch_result, chan_result = batch_axes, chan_axes
        elif (
            batch_axes == []
            or batch_axes is None
            or all(isinstance(x, int) for x in batch_axes)
        ) and (
            chan_axes == []
            or chan_axes is None
            or all(isinstance(x, int) for x in chan_axes)
        ):
            # check if it is list of indices and can make meaningful boolean arrays, if so return
            if any(i < 0 or i >= ndim for i in batch_axes) or any(
                i < 0 or i >= ndim for i in chan_axes
            ):
                raise Exception(
                    "No axes index can be smaller than zero or bigger than ndim-1!"
                )

            batch_result = [True if i in batch_axes else False for i in range(ndim)]
            chan_result = [True if i in chan_axes else False for i in range(ndim)]
        else:
            raise Exception(
                "Axes arguments has to be boolean arrays or integer index arrays!"
            )

        ndim_spacetime = ndim - (batch_result.count(True) + chan_result.count(True))
        if ndim_spacetime > 4 or ndim_spacetime < 1:
            print(batch_result)
            print(chan_result)
            print(
                ndim, batch_result.count(True), chan_result.count(True), ndim_spacetime
            )
            raise Exception(
                "Number of spacetime axes cannot be more than 4 and cannot be less than 1!"
            )

        if any(
            [batch_result[i] is True and chan_result[i] is True for i in range(ndim)]
        ):
            raise Exception("No axes can be both batch and chan axes!")

        return batch_result, chan_result
