import numpy

from aydin.features.extensible_features import ExtensibleFeatureGenerator
from aydin.features.groups.dct import DCTFeatures
from aydin.features.groups.median import MedianFeatures
from aydin.features.groups.random import RandomFeatures
from aydin.features.groups.spatial import SpatialFeatures
from aydin.features.groups.uniform import UniformFeatures
from aydin.util.log.log import lprint


class StandardFeatureGenerator(ExtensibleFeatureGenerator):
    """Standard Feature Generator"""

    def __init__(
        self,
        kernel_widths=None,
        kernel_scales=None,
        kernel_shapes=None,
        min_level: int = 0,
        max_level: int = 13,
        scale_one_width: int = 3,
        include_scale_one: bool = False,
        include_fine_features: bool = True,
        include_corner_features: bool = False,
        include_line_features: bool = False,
        decimate_large_scale_features: bool = True,
        extend_large_scale_features: bool = False,
        include_spatial_features: bool = False,
        spatial_features_coarsening: int = 2,
        num_sinusoidal_features: int = 0,
        include_median_features: bool = False,
        include_dct_features: bool = False,
        dct_max_freq: float = 0.5,
        include_random_conv_features: bool = False,
        dtype: numpy.dtype = numpy.float32,
    ):
        """Constructs a standard feature generator.

        Parameters
        ----------
        kernel_widths : numpy.typing.ArrayLike
            ArrayLike of kernel widths.
        kernel_scales : numpy.typing.ArrayLike
            ArrayLike of kernel scales.
        kernel_shapes : numpy.typing.ArrayLike
            ArrayLike of kernel shapes.
        min_level : int
            Minimum level of features to include
        max_level : int
            Maximum level of features to include
        scale_one_width : int
            Width of the scale one features
        include_scale_one : bool
            True or False, argument to set decision
            on inclusion of scale-one-features
        include_fine_features : bool
            True or False, argument to set decision
            on inclusion of fine features
        include_corner_features : bool
            True or False, argument to set decision
            on inclusion of corner features
        include_line_features : bool
            True or False, argument to set decision
            on inclusion of line features
        decimate_large_scale_features : bool
            True or False, argument to set decision
            on decimation of large scale features
        extend_large_scale_features : bool
            True or False, argument to set decision
            on extension of large scale features.
        include_spatial_features : bool
            True or False, argument to set decision
            on inclusion of spatial features
        spatial_features_coarsening : int
            Degree of coarsening to apply on spatial features
            to prevent identification of individual pixel values
        num_sinusoidal_features : int
            Number of sinusoidal features
        include_median_features : bool
            True or False, argument to set decision
            on inclusion of median features
        include_dct_features : bool
            True or False, argument to set decision
            on inclusion of dct features
        dct_max_freq : float
            Maximum included frequency during
            dct features computation.
        include_random_conv_features : bool
            True or False, argument to set decision
            on inclusion of random convolutional features
        dtype
            Datatype of the features
        """
        super().__init__()

        self.dtype = dtype
        lprint(f"Features will be computed using dtype: {dtype}")

        uniform = UniformFeatures(
            kernel_widths=kernel_widths,
            kernel_scales=kernel_scales,
            kernel_shapes=kernel_shapes,
            min_level=min_level,
            max_level=max_level,
            include_scale_one=include_scale_one,
            include_fine_features=include_fine_features,
            include_corner_features=include_corner_features,
            include_line_features=include_line_features,
            decimate_large_scale_features=decimate_large_scale_features,
            extend_large_scale_features=extend_large_scale_features,
            scale_one_width=scale_one_width,
            dtype=dtype,
        )
        self.add_feature_group(uniform)

        if include_spatial_features:
            spatial = SpatialFeatures(coarsening=spatial_features_coarsening)
            self.add_feature_group(spatial)

        if num_sinusoidal_features > 0:
            periods = list([1 / 2 ** i for i in range(num_sinusoidal_features)])
            for period in periods:
                spatial = SpatialFeatures(period=period)
                self.add_feature_group(spatial)

        if include_median_features:
            radii = [1, 2, 3]
            medians = MedianFeatures(radii=radii)
            self.add_feature_group(medians)

        if include_dct_features:
            dct3 = DCTFeatures(size=3, max_freq=dct_max_freq, power=1)
            dct5 = DCTFeatures(size=5, max_freq=dct_max_freq, power=1)
            dct7 = DCTFeatures(size=7, max_freq=dct_max_freq, power=1)
            self.add_feature_group(dct3)
            self.add_feature_group(dct5)
            self.add_feature_group(dct7)

        if include_random_conv_features:
            rnd3 = RandomFeatures(size=3, num_features=3)
            rnd5 = RandomFeatures(size=5, num_features=5)
            rnd7 = RandomFeatures(size=7, num_features=7)
            self.add_feature_group(rnd3)
            self.add_feature_group(rnd5)
            self.add_feature_group(rnd7)
