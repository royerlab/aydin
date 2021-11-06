import numpy

from aydin.features.base import FeatureGeneratorBase
from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import lprint, lsection


class ExtensibleFeatureGenerator(FeatureGeneratorBase):
    """
    Extensible Feature Generator class
    """

    def __init__(self):
        """
        Constructs an extensible feature generator
        """
        # Calls super constructor:
        super().__init__()

        # This list holds all the information for computing each 'group' of features:
        self.features_group_list = []

    def _load_internals(self, path: str):
        pass

    def add_feature_group(self, feature_group: FeatureGroupBase, *args, **kwargs):
        """
        Adds a feature to this feature generator.

        Parameters
        ----------
        feature_group : FeatureGroupBase
            feature group
        args
            additional arguments for function
        kwargs
            additional keyword arguments for function
        """
        self.features_group_list.append(feature_group)

    def clear_features(self):
        """
        Clears the features group list
        """
        self.features_group_list = []

    def get_num_features(self, ndim: int) -> int:
        """
        Returns the number of features when considering translations

        Parameters
        ----------
        ndim : int
            number of dimensions

        Returns
        -------
        nb_features : int
        """
        nb_features = 0
        for feature_group in self.features_group_list:
            nb_features += feature_group.num_features(ndim)
        return nb_features

    def get_receptive_field_radius(self) -> int:
        """
        Returns the receptive field radius in pixels

        Returns
        -------
        result : int
            receptive field radius in pixels
        """

        receptive_field_radius = 0
        for feature_group in self.features_group_list:
            receptive_field_radius = max(
                receptive_field_radius, feature_group.receptive_field_radius
            )
        return receptive_field_radius

    def compute(
        self,
        image,
        exclude_center_feature=False,
        exclude_center_value=False,
        features=None,
        feature_last_dim=True,
        passthrough_channels=None,
        num_reserved_features=0,
        excluded_voxels=None,
        spatial_feature_offset=None,
        spatial_feature_scale=None,
    ):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (n,d,h,w) where n is the number of features.

        Parameters
        ----------
        image : numpy.ndarray
        image for which features are computed
        exclude_center_feature : bool
        exclude_center_value : bool
        features
        feature_last_dim : bool
        passthrough_channels
        num_reserved_features : int
        excluded_voxels
        spatial_feature_offset
        spatial_feature_scale

        Returns
        -------
        feature array : numpy.ndarray

        """

        with lsection('Computing features'):

            # some important numbers:
            num_dims = len(image.shape)
            num_spatiotemp_dim = num_dims - 2
            num_batches = image.shape[0]
            num_channels = image.shape[1]
            num_features = self.get_num_features(num_spatiotemp_dim)

            # exclude_center_value can be a tuple, in that case each entry corresponds to a channel:
            if type(exclude_center_value) is not tuple:
                exclude_center_value = (exclude_center_value,) * num_channels

            # Fills in the default values for passthrough channels: False ==> by default channels are not passthrough.
            if passthrough_channels is None:
                passthrough_channels = (False,) * num_channels

            # Computes the number of features, taking into account: passthrough_channels, channels, and reserved_features:
            num_passthrough_channels = sum(1 if p else 0 for p in passthrough_channels)
            num_normal_features = num_features * (
                num_channels - num_passthrough_channels
            )
            num_total_features = (
                num_normal_features + num_passthrough_channels + num_reserved_features
            )

            # Creates feature array that will hold the final result:
            features = self.create_feature_array(image, num_total_features)

            # We iterate over batches:
            for batch_index in range(num_batches):

                with lsection(
                    f'Computing features for batch: {batch_index + 1}/{num_batches}'
                ):
                    feature_pointer = 0

                    # We iterate over channels:
                    for channel_index in range(num_channels):

                        with lsection(
                            f'Computing features for channel: {channel_index + 1}/{num_channels}'
                        ):
                            # We collect the 'exclude_center_value' for the current channel:
                            exclude_center_value_for_channel = exclude_center_value[
                                channel_index
                            ]

                            lprint(
                                f'Excluding center value for channel: {exclude_center_value_for_channel}'
                            )

                            # Image batch slice:
                            image_slice = (
                                batch_index,
                                channel_index,
                                *(slice(None),) * num_spatiotemp_dim,
                            )

                            lprint(f'Image slice: {image_slice}')

                            if passthrough_channels[channel_index]:
                                # A passthrough channel is simply fed directly as a feature:
                                lprint(
                                    f'Adding passthrough channel feature for channel index: {channel_index}'
                                )
                                batch_feature_slice = (
                                    slice(feature_pointer, feature_pointer + 1, 1),
                                    batch_index,
                                    *(slice(None),) * num_spatiotemp_dim,
                                )
                                features[batch_feature_slice] = image[
                                    image_slice
                                ].astype(self.dtype, copy=False)
                                feature_pointer += 1

                            else:
                                single_image = image[image_slice]

                                # Usefull code snippet for debugging features:
                                # with napari.gui_qt():
                                #      viewer = Viewer()
                                #      viewer.add_image(image_batch_gpu.get(), name='image')
                                #      viewer.add_image(rescale_intensity(image_integral_gpu.get(), in_range='image', out_range=(0, 1)), name='integral')

                                # Feature batch slice:
                                batch_feature_slice = (
                                    slice(None),
                                    batch_index,
                                    *(slice(None),) * num_spatiotemp_dim,
                                )
                                lprint(
                                    f'Feature slice for batch and channel: {batch_feature_slice}'
                                )

                                batch_channel_features = features[batch_feature_slice]

                                for feature_group in self.features_group_list:
                                    lprint(
                                        f'Computing feature {feature_group}, '  # , and kwargs={kwargs}'
                                    )

                                    # number of features in group:
                                    num_group_features = feature_group.num_features(
                                        num_spatiotemp_dim
                                    )

                                    # Excluded voxels:
                                    if excluded_voxels is None:
                                        excluded_voxels = []
                                    excluded_voxels_for_feature_group = []

                                    if exclude_center_value_for_channel:
                                        excluded_voxels_for_feature_group.extend(
                                            excluded_voxels
                                        )
                                        if (
                                            not (0,) * num_spatiotemp_dim
                                            in excluded_voxels_for_feature_group
                                        ):
                                            excluded_voxels_for_feature_group.append(
                                                (0,) * num_spatiotemp_dim
                                            )

                                    # We prepare feature generation for the group by setting the image,
                                    # any computation that can be factored should happen now:
                                    feature_group.prepare(
                                        single_image,
                                        excluded_voxels=excluded_voxels_for_feature_group,
                                        offset=spatial_feature_offset,
                                        scale=spatial_feature_scale,
                                    )

                                    for index in range(num_group_features):
                                        # feature array
                                        feature = batch_channel_features[
                                            feature_pointer
                                        ]

                                        # Computing the feature:
                                        feature_group.compute_feature(index, feature)

                                        # Increments feature index:
                                        feature_pointer += 1

                                    feature_group.finish()

            # # Usefull code snippet for debugging features:
            # import napari
            #
            # with napari.gui_qt():
            #     from napari import Viewer
            #
            #     viewer = Viewer()
            #     viewer.add_image(features, name='features')

            # 'collect_features_nD' puts the feature vector in axis 0.
            # The following line creates a view of the array
            # in which the features are indexed by the last dimension instead:
            if feature_last_dim:
                lprint('Move feature axis to the last axis ...')
                features = numpy.moveaxis(features, 0, -1)

            return features
