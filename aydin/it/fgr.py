import gc
import time
from typing import Optional
import numpy

from aydin.features.base import FeatureGeneratorBase
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.it.balancing.data_histogram_balancer import DataHistogramBalancer
from aydin.it.base import ImageTranslatorBase
from aydin.regression.base import RegressorBase
from aydin.regression.cb import CBRegressor
from aydin.util.array.nd import nd_split_slices
from aydin.util.log.log import lprint, lsection
from aydin.util.offcore.offcore import offcore_array


class ImageTranslatorFGR(ImageTranslatorBase):
    """
    Feature Generation & Regression (FGR) based Image TranslatorFGR Image Translator
    """

    feature_generator: FeatureGeneratorBase

    def __init__(
        self,
        feature_generator: FeatureGeneratorBase = None,
        regressor: RegressorBase = None,
        balance_training_data: bool = False,
        voxel_keep_ratio: float = 1,
        max_voxels_for_training: Optional[int] = None,
        favour_bright_pixels: bool = False,
        **kwargs,
    ):
        """Constructs a FGR image translator. FGR image translators use feature generation
        and regression learning to acheive image translation.

        Parameters
        ----------
        feature_generator : FeatureGeneratorBase
            Feature generator.
        regressor : RegressorBase
            Regressor.
        balance_training_data : bool
            Limits number training entries per target
            value histogram bin.
        voxel_keep_ratio : float
            Ratio of the voxels to keep for training.
        max_voxels_for_training : int, optional
            Maximum number of the voxels that can be
            used for training.
        favour_bright_pixels : bool
            Marks bright pixels more favourable for
            the data histogram balancer.
        kwargs : dict
            Keyword arguments.
        max_memory_usage_ratio : float
            Maximum allowed memory load.
        max_tiling_overhead : float
            Maximum allowed margin overhead during tiling.
        """
        super().__init__(**kwargs)

        self.voxel_keep_ratio = voxel_keep_ratio
        self.balance_training_data = balance_training_data
        self.favour_bright_pixels = favour_bright_pixels
        self.feature_generator = (
            StandardFeatureGenerator()
            if feature_generator is None
            else feature_generator
        )
        self.regressor = CBRegressor() if regressor is None else regressor

        self.max_voxels_for_training = (
            self.regressor.recommended_max_num_datapoints()
            if max_voxels_for_training is None
            else max_voxels_for_training
        )

        # It seems empirically that excluding the central feature incurs no cost in quality,
        # simply because for every scale the next scale already partly covers these pixels.
        self.exclude_center_feature = True

        # Option to exclude the center value during translation. Set to True by default.
        self.exclude_center_value_when_translating = False

        # Advanced private functionality:
        # This field gives the opportunity to specify which channels must be 'passed-through'
        # directly as features. This is not intended to be used directly by users.
        self._passthrough_channels = None

        with lsection("FGR image translator"):
            lprint(f"balance training data: {self.balance_training_data}")

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
        with lsection(f"Saving 'fgr' image translator to {path}"):
            frozen = super().save(path)
            frozen += self.feature_generator.save(path) + '\n'
            frozen += self.regressor.save(path) + '\n'

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'fgr' image translator from {path}"):
            self.feature_generator = FeatureGeneratorBase.load(path)
            self.regressor = RegressorBase.load(path)

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['feature_generator']
        del state['regressor']
        return state

    def stop_training(self):
        """Stops currently running training within the instance by calling
        the corresponding `stop_fit()` method on the regressor.
        """
        self.regressor.stop_fit()

    def _estimate_memory_needed_and_available(self, image):
        with lsection(
            "Estimating the amount of memory needed to store feature arrays:"
        ):
            num_spatio_temp_dim = len(image.shape[2:])
            dummy_image = numpy.zeros(
                shape=(1, 1) + (4,) * num_spatio_temp_dim, dtype=numpy.float32
            )

            x = self._compute_features(
                dummy_image,
                exclude_center_feature=self.exclude_center_feature,
                exclude_center_value=False,
            )
            num_features = x.shape[-1]
            feature_dtype = x.dtype

            (
                memory_needed,
                memory_available,
            ) = super()._estimate_memory_needed_and_available(image)

            """Here, 1.3 correction factor is chosen to have a bigger safety margin as we use
            more memory while generating features(namely uniform features) than the amount
            of memory we need after feature computations."""
            memory_needed = max(
                memory_needed, 1.3 * num_features * image.size * feature_dtype.itemsize
            )

            return memory_needed, memory_available

    def _compute_features(
        self,
        image,
        exclude_center_feature,
        exclude_center_value,
        features_last=True,
        num_reserved_features=0,
        image_slice=None,
        whole_image_shape=None,
    ):
        """Internal function that computes features for a given image.

        :param image: image
        :param exclude_center_feature: exclude center feature
        :param exclude_center_value: exclude center value
        :return: returns flattened array of features
        """

        with lsection(f"Computing features for image of shape {image.shape}:"):
            excluded_voxels = (
                None
                if self.blind_spots is None
                else list(
                    [
                        coordinate
                        for coordinate in self.blind_spots
                        if coordinate != (0,) * (image.ndim - 2)
                    ]
                )
            )

            lprint(f"exclude_center_feature = {exclude_center_feature}")
            lprint(f"exclude_center_value   = {exclude_center_value}")
            lprint(f"excluded_voxels        = {excluded_voxels}")

            excluded_voxels = (
                None
                if self.blind_spots is None
                else list(
                    [
                        coordinate
                        for coordinate in self.blind_spots
                        if coordinate != (0,) * (image.ndim - 2)
                    ]
                )
            )

            # If this is a part of a larger image, we can figure out what are the offsets and scales for the spatial features:
            spatial_feature_scale = (
                None
                if whole_image_shape is None
                else tuple(1.0 / s for s in whole_image_shape[2:])
            )
            spatial_feature_offset = (
                None if image_slice is None else tuple(s.start for s in image_slice[2:])
            )
            lprint(f"spatial_feature_scale     = {spatial_feature_scale}")
            lprint(f"spatial_feature_offset    = {spatial_feature_offset}")

            features = self.feature_generator.compute(
                image,
                exclude_center_feature=exclude_center_feature,
                exclude_center_value=exclude_center_value,
                num_reserved_features=num_reserved_features,
                passthrough_channels=self._passthrough_channels,
                feature_last_dim=False,
                excluded_voxels=excluded_voxels,
                spatial_feature_offset=spatial_feature_offset,
                spatial_feature_scale=spatial_feature_scale,
            )

            num_features = features.shape[0]
            x = features.reshape(num_features, -1)

            if features_last:
                x = numpy.moveaxis(x, 0, -1)

            return x

    def _train(
        self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        with lsection(
            f"Training image translator from image of shape {input_image.shape} to image of shape {target_image.shape}:"
        ):

            self.prepare_monitoring_images()

            num_input_channels = input_image.shape[1]
            num_target_channels = target_image.shape[1]
            normalised_input_shape = input_image.shape

            # Deal with jinv parameter;
            if jinv is None:
                exclude_center_value = self.self_supervised
            elif type(jinv) is tuple:
                exclude_center_value = jinv
            else:
                exclude_center_value = jinv

            # Prepare the splitting of train from valid data as well as balancing and decimation...
            self._prepare_split_train_val(input_image, target_image)

            # Tilling strategy is determined here:
            tilling_strategy, margins = self._get_tilling_strategy_and_margins(
                input_image,
                max_voxels_per_tile=self.max_voxels_per_tile,
                min_margin=self.tile_min_margin,
                max_margin=self.tile_max_margin,
            )
            lprint(f"Tilling strategy (just batches): {tilling_strategy}")
            lprint(f"Margins for tiles: {margins} .")

            # tile slice objects with margins:
            tile_slices_margins = list(
                nd_split_slices(
                    normalised_input_shape, nb_slices=tilling_strategy, margins=margins
                )
            )

            # Number of tiles:
            number_of_tiles = len(tile_slices_margins)
            lprint(f"Number of tiles (slices): {number_of_tiles}")

            # We initialise the arrays:
            x_train, x_valid, y_train, y_valid = (None,) * 4

            for idx, slice_margin_tuple in enumerate(tile_slices_margins):
                with lsection(
                    f"Current tile: {idx}/{number_of_tiles}, slice: {slice_margin_tuple} "
                ):

                    # We first extract the tile image:
                    input_image_tile = input_image[slice_margin_tuple]
                    target_image_tile = target_image[slice_margin_tuple]

                    x_tile = self._compute_features(
                        input_image_tile,
                        exclude_center_feature=self.exclude_center_feature,
                        exclude_center_value=exclude_center_value,
                        num_reserved_features=0,
                        features_last=False,
                        image_slice=slice_margin_tuple,
                        whole_image_shape=input_image.shape,
                    )
                    y_tile = target_image_tile.reshape(num_target_channels, -1)

                    num_features_tile = x_tile.shape[0]
                    num_entries_tile = y_tile.shape[1]

                    lprint(
                        f"Number of entries: {num_entries_tile}, features: {num_features_tile}, input channels: {num_input_channels}, target channels: {num_target_channels}"
                    )

                    # We split this tile's data into train and valid sets:
                    (
                        x_train_tile,
                        x_valid_tile,
                        y_train_tile,
                        y_valid_tile,
                    ) = self._do_split_train_val(
                        num_target_channels, train_valid_ratio, x_tile, y_tile
                    )

                    # We get rid of x and y to free memory:
                    del x_tile, y_tile
                    gc.collect()

                    # We now put the feature dimension to the back:
                    x_train_tile = numpy.moveaxis(x_train_tile, 0, -1)
                    x_valid_tile = numpy.moveaxis(x_valid_tile, 0, -1)

                    if x_train is None or x_valid is None:
                        x_train, x_valid, y_train, y_valid = (
                            x_train_tile,
                            x_valid_tile,
                            y_train_tile,
                            y_valid_tile,
                        )
                    else:
                        x_train = numpy.append(x_train, x_train_tile, axis=0)
                        x_valid = numpy.append(x_valid, x_valid_tile, axis=0)
                        y_train = numpy.append(y_train, y_train_tile, axis=1)
                        y_valid = numpy.append(y_valid, y_valid_tile, axis=1)

            lprint("Training now...")
            self.regressor.fit(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                regressor_callback=self.get_callback() if self.monitor else None,
            )

            self.loss_history = self.regressor.loss_history

    def prepare_monitoring_images(self):
        # Compute features for monitoring images:
        if self.monitor is not None and self.monitor.monitoring_images is not None:
            # Normalise monitoring images:
            normalised_monitoring_images = [
                self.shape_normaliser.normalise(
                    monitoring_image, batch_dims=None, channel_dims=None
                )
                for monitoring_image in self.monitor.monitoring_images
            ]

            # compute features proper:
            monitoring_images_features = [
                self._compute_features(
                    monitoring_image,
                    exclude_center_feature=self.exclude_center_feature,
                    exclude_center_value=False,
                    features_last=True,
                )
                for monitoring_image in normalised_monitoring_images
            ]
        else:
            monitoring_images_features = None
        # We keep these features handy...
        self.monitoring_datasets = monitoring_images_features

    def get_callback(self):
        # Regressor callback:
        def regressor_callback(iteration, val_loss, model):

            if val_loss is None:
                return

            current_time_sec = time.time()

            # Correct for dtype range:
            if self.feature_generator.dtype == numpy.uint8:
                val_loss /= 255
            elif self.feature_generator.dtype == numpy.uint16:
                val_loss /= 255 * 255

            if current_time_sec > self.last_callback_time_sec + self.callback_period:

                if self.monitoring_datasets and self.monitor:
                    predicted_monitoring_datasets = [
                        self.regressor.predict(x_m, models_to_use=[model])
                        for x_m in self.monitoring_datasets
                    ]
                    inferred_images = [
                        y_m.reshape(image.shape)
                        for (image, y_m) in zip(
                            self.monitor.monitoring_images,
                            predicted_monitoring_datasets,
                        )
                    ]

                    # for image in inferred_images:
                    for index in range(len(inferred_images)):
                        (
                            inferred_images[index],
                            _,
                            _,
                        ) = self.shape_normaliser.shape_normalize(
                            inferred_images[index]
                        )

                    denormalised_inferred_images = [
                        self.target_shape_normaliser.denormalise(inferred_image)
                        for inferred_image in inferred_images
                    ]

                    self.monitor.variables = (
                        iteration,
                        val_loss,
                        denormalised_inferred_images,
                    )
                elif self.monitor:
                    self.monitor.variables = (iteration, val_loss, None)

                self.last_callback_time_sec = current_time_sec
            else:
                pass
                # print(f"Iteration={iteration} metric value: {eval_metric_value} ")

        return regressor_callback

    def _prepare_split_train_val(self, input_image, target_image):
        # the number of voxels:
        num_of_voxels = input_image.size
        lprint(
            f"Image has: {num_of_voxels} voxels, at most: {self.max_voxels_for_training} voxels will be used for training or validation."
        )
        # This is the ratio of pixels to keep:
        max_voxels_keep_ratio = float(self.max_voxels_for_training) / num_of_voxels
        effective_keep_ratio = min(self.voxel_keep_ratio, max_voxels_keep_ratio)
        lprint(
            f"Given train ratio is: {self.voxel_keep_ratio}, max_voxels induced keep-ratio is: {max_voxels_keep_ratio}"
        )

        # For small images it is not worth having any limit or balance anything:
        num_voxels_in_small_image = 5 * 1e6
        effective_keep_ratio = (
            1.0 if num_of_voxels < num_voxels_in_small_image else effective_keep_ratio
        )
        balance_training_data = (
            False
            if num_of_voxels < num_voxels_in_small_image
            else self.balance_training_data
        )

        lprint(
            f"Data histogram balancer is: {'active' if balance_training_data else 'inactive'}"
        )
        lprint(f"Effective keep-ratio is: {effective_keep_ratio}")
        lprint(
            f"Favouring bright pixels: {'yes' if self.favour_bright_pixels else 'no'}"
        )

        # We decide on a 'batch' length that will be used to shuffle, select and then copy the training data...
        num_of_voxels_per_stack = input_image[2:].size
        batch_length = (
            16
            if num_of_voxels_per_stack < 1e5
            else (
                32
                if num_of_voxels_per_stack < 1e6
                else (
                    128
                    if num_of_voxels_per_stack < 1e7
                    else (512 if num_of_voxels_per_stack < 1e8 else 4096)
                )
            )
        )
        lprint(f"Using contiguous batches of length: {batch_length} ")
        # We create a balancer common to all tiles:
        balancer = DataHistogramBalancer(
            balance=balance_training_data,
            keep_ratio=effective_keep_ratio,
            favour_bright_pixels=self.favour_bright_pixels,
        )
        # Calibration of the balancer is done on the entire image:
        lprint("Calibrating balancer...")
        balancer.calibrate(target_image.ravel(), batch_length)

        # Keep both balancer and batch length
        self.train_val_split_balancer = balancer
        self.train_val_split_batch_length = batch_length

    def _do_split_train_val(
        self, num_target_channels: int, train_valid_ratio: float, x, y
    ):
        with lsection(
            f"Splitting train and test sets (train_test_ratio={train_valid_ratio}) "
        ):
            balancer: DataHistogramBalancer = self.train_val_split_balancer
            batch_length: int = self.train_val_split_batch_length

            nb_features = x.shape[0]
            nb_entries = y.shape[1]

            nb_split_batches = max(nb_entries // batch_length, 64)
            lprint(
                f"Creating random indices for train/val split (nb_split_batches={nb_split_batches})"
            )

            nb_split_batches_valid = int(train_valid_ratio * nb_split_batches)
            nb_split_batches_train = nb_split_batches - nb_split_batches_valid
            is_train_array = numpy.full(nb_split_batches, False)
            is_train_array[nb_split_batches_valid:] = True
            lprint(f"train/valid bool array created (length={is_train_array.shape[0]})")

            lprint("Shuffling train/valid bool array...")
            numpy.random.shuffle(is_train_array)

            lprint("Calculating number of entries for train and validation...")
            nb_entries_per_split_batch = max(1, nb_entries // nb_split_batches)
            nb_entries_train = nb_split_batches_train * nb_entries_per_split_batch
            nb_entries_valid = nb_split_batches_valid * nb_entries_per_split_batch

            lprint(
                f"Number of entries for training: {nb_entries_train} = {nb_split_batches_train}*{nb_entries_per_split_batch}, validation: {nb_entries_valid} = {nb_split_batches_valid} * {nb_entries_per_split_batch}"
            )

            lprint("Allocating arrays...")
            x_train = offcore_array(
                shape=(nb_features, nb_entries_train),
                dtype=x.dtype,
                max_memory_usage_ratio=self.max_memory_usage_ratio,
            )
            y_train = offcore_array(
                shape=(num_target_channels, nb_entries_train),
                dtype=y.dtype,
                max_memory_usage_ratio=self.max_memory_usage_ratio,
            )
            x_valid = offcore_array(
                shape=(nb_features, nb_entries_valid),
                dtype=x.dtype,
                max_memory_usage_ratio=self.max_memory_usage_ratio,
            )
            y_valid = offcore_array(
                shape=(num_target_channels, nb_entries_valid),
                dtype=y.dtype,
                max_memory_usage_ratio=self.max_memory_usage_ratio,
            )

            with lsection("Copying data for training and validation sets..."):

                # We use a random permutation to avoid having the balancer drop only from the 'end' of the image
                permutation = numpy.random.permutation(nb_split_batches)

                i, jt, jv = 0, 0, 0
                dst_stop_train = 0
                dst_stop_valid = 0

                balancer.initialise(nb_split_batches)

                for is_train in numpy.nditer(is_train_array):
                    if i % (nb_split_batches // 64) == 0:
                        lprint(
                            f"Copying section [{i},{min(nb_split_batches, i + nb_split_batches // 64)}]"
                        )

                    permutated_i = permutation[i]
                    src_start = permutated_i * nb_entries_per_split_batch
                    src_stop = src_start + nb_entries_per_split_batch
                    i += 1

                    xsrc = x[:, src_start:src_stop]
                    ysrc = y[:, src_start:src_stop]

                    if balancer.add_entry(ysrc):
                        if is_train:
                            dst_start_train = jt * nb_entries_per_split_batch
                            dst_stop_train = (jt + 1) * nb_entries_per_split_batch

                            jt += 1

                            xdst = x_train[:, dst_start_train:dst_stop_train]
                            ydst = y_train[:, dst_start_train:dst_stop_train]

                            numpy.copyto(xdst, xsrc)
                            numpy.copyto(ydst, ysrc)

                        else:
                            dst_start_valid = jv * nb_entries_per_split_batch
                            dst_stop_valid = (jv + 1) * nb_entries_per_split_batch

                            jv += 1

                            xdst = x_valid[:, dst_start_valid:dst_stop_valid]
                            ydst = y_valid[:, dst_start_valid:dst_stop_valid]

                            numpy.copyto(xdst, xsrc)
                            numpy.copyto(ydst, ysrc)

                # Now we actually truncate out all the zeros at the end of the arrays:
                x_train = x_train[:, 0:dst_stop_train]
                y_train = y_train[:, 0:dst_stop_train]
                x_valid = x_valid[:, 0:dst_stop_valid]
                y_valid = y_valid[:, 0:dst_stop_valid]

                lprint(f"Histogram all    : {balancer.get_histogram_all_as_string()}")
                lprint(f"Histogram kept   : {balancer.get_histogram_kept_as_string()}")
                lprint(
                    f"Histogram dropped: {balancer.get_histogram_dropped_as_string()}"
                )
                lprint(
                    f"Number of entries kept: {balancer.total_kept()} out of {balancer.total_entries} total"
                )
                lprint(
                    f"Percentage of data kept: {100 * balancer.percentage_kept():.3f}% (train_data_ratio={balancer.keep_ratio}) "
                )
                if balancer.keep_ratio >= 1 and balancer.percentage_kept() < 1:
                    lprint(
                        "Note: balancer has dropped entries that fell on over-represented histogram bins"
                    )
        return x_train, x_valid, y_train, y_valid

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Internal method that translates an input image on the basis of the trained model.

        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        shape = input_image.shape
        num_batches = shape[0]

        x = self._compute_features(
            input_image,
            exclude_center_feature=self.exclude_center_feature,
            exclude_center_value=self.exclude_center_value_when_translating,
            num_reserved_features=0,
            features_last=True,
            image_slice=image_slice,
            whole_image_shape=whole_image_shape,
        )

        with lsection(
            f"Predict from feature vector of dimension {x.shape} and dtype: {x.dtype}:"
        ):
            lprint("Predicting... ")
            # Predict using regressor:
            yp = self.regressor.predict(x)

            # We make sure that we have the result in float type, but make _sure_ to avoid copying data:
            if yp.dtype != numpy.float32 and yp.dtype != numpy.float64:
                yp = yp.astype(numpy.float32, copy=False)

            # We reshape the array:
            num_target_channels = yp.shape[0]
            translated_image_shape = (num_batches, num_target_channels) + shape[2:]
            lprint(f"Reshaping array to {translated_image_shape}... ")
            inferred_image = yp.reshape(translated_image_shape)

        return inferred_image
