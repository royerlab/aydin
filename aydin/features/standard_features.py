import numpy

from aydin.features.extensible_features import ExtensibleFeatureGenerator
from aydin.features.groups.dct import DCTFeatures
from aydin.features.groups.lowpass import LowPassFeatures
from aydin.features.groups.median import MedianFeatures
from aydin.features.groups.random import RandomFeatures
from aydin.features.groups.spatial import SpatialFeatures
from aydin.features.groups.uniform import UniformFeatures
from aydin.util.log.log import lprint


class StandardFeatureGenerator(ExtensibleFeatureGenerator):
    """
    The standard feature generator provides the following set of features:
    multiscale integral(uniform) features of different shapes, spatial
    features, median features, dct features, deterministic random
    convolutional features.

    """

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
        include_lowpass_features: bool = True,
        num_lowpass_features: int = 8,
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
            Minimum scale level of features to include
            (advanced)

        max_level : int
            Maximum scale level of features to include

        scale_one_width : int
            Width of the scale one features.
            (advanced)

        include_scale_one : bool
            When True scale-one-features are included. Uniform scale-one
            features consist in simply passing the intensity values of pixels
            in direct proximity to the center pixel. These features encode
            high-frequency information that might be heavily contaminated by
            noise, so use with caution. We recommend using this only for
            moderate noise levels, or for images where strong high-frequency
            signal is present and needs to be recovered.

        include_fine_features : bool
            When True fine features are included. Uniform fine features
            consist in summing up pixel values over small groups of 2 or 3
            pixels surrounding the center pixel. These features encode higher
            frequency information than other features (only scale-one feature
            are even higher frequency).

        include_corner_features : bool
            When True corner features are included. Corner features are
            uniform features that consists in summing the intensity values of
            groups of pixels along the corners of the typical default
            multi-scale features.

        include_line_features : bool
            When True line features are included. Line features are another
            flavour of uniform features that consist in summing up the pixel
            intensity values along one-pixel-wide lines around the center
            pixel.
            (advanced)

        decimate_large_scale_features : bool
            When True large scale features are decimated. To reduce the number
            of features it can be advantageous to reduce the number of
            large-scale (low-freq) features by decimating them. This is done
            by removing center features that overlap with already covered
            features at lower scales.
            (advanced)

        extend_large_scale_features : bool
            When True large scale features are extended. Extending large
            scale features makes these feature cover more pixels by
            overlapping pixels at the center of the receptive field.
            (advanced)

        include_spatial_features : bool
            When True spatial features are included. Spatial features are simply
            the shifted, scaled, and possibly quantised coordinates of the voxels
            themselves. This should only be used if you train on the whole image
            that you intend to process. If applied on other images than the one
            trained on, ensure that any spatial bias learned such as the position
            of certain image structures or degradation over space is consistent
            between the image you trained on and the images you process.

        spatial_features_coarsening : int
            Degree of coarsening to apply on spatial features
            to prevent identification of individual pixel values.
            (advanced)

        num_sinusoidal_features : int
            Number of sinusoidal features to include. Sinusoidal features are
            spatial features that are sinusoidal.
            (advanced)

        include_median_features : bool
            When True median features are included. Median features consist
            in the median-filtered image
            with kernels of sizes 3^n, 5^n and 7^n.
            (advanced)

        include_lowpass_features : bool
            Includes lowpass image features that are computed by applying
            Butterworth filters at regular frequency intervals. Highly effective
            for images for which Butterworth denoising also works well. However,
            for large dimensions with many dimensions feature generation can be
            slow.

        num_lowpass_features : int
            Number of lowpass features to include.
            (advanced)

        include_dct_features : bool
            When True DCT features computed on per-voxel-centered image patches
            are included.
            (advanced)

        dct_max_freq : float
            Maximum included frequency during dct features computation.
            Should be a number within [0, 1]
            (advanced)

        include_random_conv_features : bool
            When True random convolutional features are included.
            This is experimental and of academic interest only.
            (advanced)


        dtype
            Datatype of the features
            (advanced)
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

        if include_lowpass_features:
            lowpass = LowPassFeatures(num_features=num_lowpass_features)
            self.add_feature_group(lowpass)
