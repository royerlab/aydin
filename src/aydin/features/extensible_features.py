"""Extensible feature generator that composes multiple feature groups."""

from typing import List, Optional, Tuple

import numpy
from numpy import ndarray

from aydin.features.base import FeatureGeneratorBase
from aydin.features.groups.base import FeatureGroupBase
from aydin.util.log.log import aprint, asection


class ExtensibleFeatureGenerator(FeatureGeneratorBase):
    """Feature generator that composes multiple feature groups.

    This generator allows building a feature set by adding multiple
    ``FeatureGroupBase`` instances. Each feature group contributes its own
    set of features, and they are all computed sequentially during the
    ``compute`` call.
    <notgui>
    """

    def __init__(self):
        """Construct an extensible feature generator with an empty group list."""
        # Calls super constructor:
        super().__init__()

        # This list holds all the information for computing each 'group' of features:
        self.features_group_list = []

    def _load_internals(self, path: str):
        """Load internal state (no-op for this class).

        Parameters
        ----------
        path : str
            Directory path (unused).
        """
        pass

    def add_feature_group(self, feature_group: FeatureGroupBase, *args, **kwargs):
        """Add a feature group to this generator.

        Parameters
        ----------
        feature_group : FeatureGroupBase
            Feature group to add.
        args
            Additional positional arguments (reserved for future use).
        kwargs
            Additional keyword arguments (reserved for future use).
        """
        self.features_group_list.append(feature_group)

    def clear_features(self):
        """Remove all feature groups from this generator."""
        self.features_group_list = []

    def get_num_features(self, ndim: int) -> int:
        """Return the total number of features across all feature groups.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the image.

        Returns
        -------
        nb_features : int
            Total number of features produced by all groups combined.
        """
        nb_features = 0
        for feature_group in self.features_group_list:
            nb_features += feature_group.num_features(ndim)
        return nb_features

    def get_receptive_field_radius(self) -> int:
        """Return the maximum receptive field radius across all feature groups.

        Returns
        -------
        receptive_field_radius : int
            Maximum receptive field radius in pixels across all groups.
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
        exclude_center_feature: bool = False,
        exclude_center_value: bool = False,
        features: ndarray = None,
        feature_last_dim: bool = True,
        passthrough_channels: Optional[Tuple[bool]] = None,
        num_reserved_features: int = 0,
        excluded_voxels: Optional[List[Tuple[int]]] = None,
        spatial_feature_offset: Optional[Tuple[float, ...]] = None,
        spatial_feature_scale: Optional[Tuple[float, ...]] = None,
    ):
        """
        Compute features for the given image using all registered feature groups.

        Iterates over batches and channels, computing features from each
        feature group and assembling them into a single output array.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which features are computed. Expected to be in standard
            form with shape ``(batch, channel, *spatial_dims)``.

        exclude_center_feature : bool
            If True, features that use the image patch's center pixel are
            entirely excluded from the set of computed features.

        exclude_center_value : bool
            If True, the center pixel is never used to compute any feature.
            Different feature generation algorithms can take different
            approaches to achieve that.

        features : ndarray
            If None the feature array is allocated internally.
            If not None the provided array is used to store the features.

        feature_last_dim : bool
            If True the last dimension of the feature array is the feature
            dimension; if False then it is the first dimension.

        passthrough_channels : Optional[Tuple[bool]]
            Optional tuple of booleans that specify which channels are
            'pass-through' channels, i.e. channels that are not featurised
            and directly used as features.

        num_reserved_features : int
            Number of features to be left blank, useful when adding
            features separately.

        excluded_voxels : Optional[List[Tuple[int]]]
            List of pixel coordinates -- expressed as tuples of ints relative
            to the central pixel -- that will be excluded from any computed
            features. This is used for implementing 'extended blind-spot'
            Noise2Self denoising approaches.

        spatial_feature_offset : Optional[Tuple[float, ...]]
            Offset vector to be applied (added) to the spatial features
            (if used).

        spatial_feature_scale : Optional[Tuple[float, ...]]
            Scale vector to be applied (multiplied) to the spatial features
            (if used).

        Returns
        -------
        features : numpy.ndarray
            The computed feature array.
        """

        with asection('Computing features'):

            # some important numbers:
            num_dims = len(image.shape)
            num_spatiotemp_dim = num_dims - 2
            num_batches = image.shape[0]
            num_channels = image.shape[1]
            num_features = self.get_num_features(num_spatiotemp_dim)

            # exclude_center_value can be a tuple, in that case
            # each entry corresponds to a channel:
            if type(exclude_center_value) is not tuple:
                exclude_center_value = (exclude_center_value,) * num_channels

            # Fills in the default values for passthrough channels:
            # False ==> by default channels are not passthrough.
            if passthrough_channels is None:
                passthrough_channels = (False,) * num_channels

            # Computes the number of features, taking into account:
            # passthrough_channels, channels, and reserved_features:
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

                with asection(
                    f'Computing features for batch: {batch_index + 1}/{num_batches}'
                ):
                    feature_pointer = 0

                    # We iterate over channels:
                    for channel_index in range(num_channels):

                        with asection(
                            f'Computing features for channel: '
                            f'{channel_index + 1}/{num_channels}'
                        ):
                            # We collect the 'exclude_center_value'
                            # for the current channel:
                            exclude_center_value_for_channel = exclude_center_value[
                                channel_index
                            ]

                            aprint(
                                f'Excluding center value for '
                                f'channel: {exclude_center_value_for_channel}'
                            )

                            # Image batch slice:
                            image_slice = (
                                batch_index,
                                channel_index,
                                *(slice(None),) * num_spatiotemp_dim,
                            )

                            aprint(f'Image slice: {image_slice}')

                            if passthrough_channels[channel_index]:
                                # A passthrough channel is simply
                                # fed directly as a feature:
                                aprint(
                                    f'Adding passthrough channel '
                                    f'feature for channel '
                                    f'index: {channel_index}'
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

                                # Useful code snippet for debugging features:
                                # with napari.gui_qt():
                                #      viewer = Viewer()
                                #      viewer.add_image(image_batch_gpu.get(), name='image')  # noqa: E501
                                #      viewer.add_image(rescale_intensity(image_integral_gpu.get(), in_range='image', out_range=(0, 1)), name='integral')  # noqa: E501

                                # Feature batch slice:
                                batch_feature_slice = (
                                    slice(None),
                                    batch_index,
                                    *(slice(None),) * num_spatiotemp_dim,
                                )
                                aprint(
                                    f'Feature slice for batch and '
                                    f'channel: {batch_feature_slice}'
                                )

                                batch_channel_features = features[batch_feature_slice]

                                for feature_group in self.features_group_list:
                                    aprint(
                                        f'Computing feature '
                                        f'{feature_group}, '  # , and kwargs={kwargs}'
                                    )

                                    # number of features in group:
                                    num_group_features = feature_group.num_features(
                                        num_spatiotemp_dim
                                    )

                                    # Excluded voxels:
                                    if excluded_voxels is None:
                                        excluded_voxels = []
                                    else:
                                        excluded_voxels = list(excluded_voxels)
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

                                    # We prepare feature generation for
                                    # the group by setting the image,
                                    # any computation that can be
                                    # factored should happen now:
                                    feature_group.prepare(
                                        single_image,
                                        excluded_voxels=(
                                            excluded_voxels_for_feature_group
                                        ),
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

            # # Useful code snippet for debugging features:
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
                aprint('Move feature axis to the last axis ...')
                features = numpy.moveaxis(features, 0, -1)

            return features
