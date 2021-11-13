from functools import reduce
from operator import mul
from typing import Tuple, Union, Sequence

import numexpr
import numpy
from numba import jit, cuda
from numba.cuda import CudaSupportError

from aydin.features.groups.base import FeatureGroupBase


from aydin.util.array.nd import nd_range_radii
from aydin.util.fast_shift.fast_shift import fast_shift
from aydin.util.fast_uniform_filter.numba_cpu_uf import numba_cpu_uniform_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.util.log.log import lprint


# Removes duplicates without changing list's order:
def _remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class UniformFeatures(FeatureGroupBase):
    """
    Uniform Feature Group class
    """

    def __init__(
        self,
        kernel_widths=None,
        kernel_scales=None,
        kernel_shapes=None,
        min_level=0,
        max_level=13,
        include_scale_one=False,
        include_fine_features=True,
        include_corner_features=False,
        include_line_features=False,
        decimate_large_scale_features=True,
        extend_large_scale_features=False,
        scale_one_width=3,
        dtype=numpy.float32,
    ):
        super().__init__()

        # Setting up default features:
        #  0,  1,  2,  3,  4,  5,  6,   7,   8,    9,   10,   11
        if kernel_widths is None:
            kernel_widths = []
            if include_scale_one:
                kernel_widths += [scale_one_width]
            if include_fine_features:
                kernel_widths += [3, 3, 3]
            if include_line_features:
                kernel_widths += [3, 3, 3]
            kernel_widths += [3] * 10
        if kernel_scales is None:
            kernel_scales = []
            if include_scale_one:
                kernel_scales += [1]
            if include_fine_features:
                kernel_scales += [3, 5, 7]
            if include_line_features:
                kernel_scales += [3, 5, 7]
            kernel_scales += [2 ** i - 1 for i in range(2, 12)]
        if kernel_shapes is None:
            kernel_shapes = []
            if include_scale_one:
                kernel_shapes += ['li']
            if include_fine_features:
                special_feature_shape = '#linc' if include_corner_features else '#l1nc'
                kernel_shapes += [special_feature_shape] * 3
            if include_line_features:
                kernel_shapes += ['|l1'] * 3

            prefix = '*' if extend_large_scale_features else ''

            kernel_shapes += [prefix + 'l2'] * 3

            if decimate_large_scale_features:
                kernel_shapes += [prefix + 'l1nc'] * 4 + [prefix + 'l1oc'] * 3
            else:
                kernel_shapes += [prefix + 'l1'] * 4 + [prefix + 'l1'] * 3

        self.kernel_widths = kernel_widths
        self.kernel_scales = kernel_scales
        self.kernel_shapes = kernel_shapes

        self.min_level = min_level
        self.max_level = max_level

        self.dtype = dtype

        self._feature_descriptions_list = None
        self._size_to_full_feature = None

        self.image = None
        self.gpu_image = None
        self.cuda_stream = None
        self.original_dtype = None
        self.excluded_voxels: Sequence[Tuple[int, ...]] = []
        self.args = None
        self.kwargs = None

    def _ensure_feature_description_available(self, ndim: int):
        if (
            self._feature_descriptions_list is None
            or len(self._feature_descriptions_list[0][0]) != ndim
        ):
            self._feature_descriptions_list = self._get_feature_descriptions_list(ndim)

    def _get_feature_descriptions_list(self, ndim: int):
        """
        Get feature descriptions

        Parameters
        ----------
        ndim : int

        Returns
        -------
        feature_description_list

        """
        feature_description_list = []

        level = 0
        for width, scale, shape in zip(
            self.kernel_widths, self.kernel_scales, self.kernel_shapes
        ):
            # Check if we have passed the max number of features already:
            # Important: We might overshoot a bit, but that's ok to make sure we get all features at a given scale...
            if level >= self.max_level:
                break
            elif level < self.min_level:
                level += 1
                continue
            level += 1

            # Computing the radius:
            radius = width // 2

            # We compute the radii along the different dimensions:
            radii = list((max(1, radius),) * ndim)

            # We generate all feature shift vectors:
            features_shifts = list(nd_range_radii(radii))

            # print(f'Feature shifts: {features_shifts}')

            # For each feature shift we append to the feature description list:
            for feature_shift in features_shifts:

                # Excluding the center pixel/feature:
                if scale == 1 and feature_shift == (0,) * ndim:
                    continue

                # if scale == 2 and width == 1:
                #     effective_shift = feature_shift
                #     negative_extent = (0,) * ndim
                #     positive_extent = (2,) * ndim

                # Different 'shapes' of feature  distributions:
                if 'l1' in shape and sum([abs(i) for i in feature_shift]) > radius:
                    continue
                elif (
                    'l2' in shape
                    and sum([i * i for i in feature_shift]) > radius * radius
                ):
                    continue
                elif 'li' in shape:
                    pass

                # keep only center (oc) or remove all that are center (nc)
                if 'oc' in shape and sum([abs(i) for i in feature_shift]) > 0:
                    continue
                elif 'nc' in shape and sum([abs(i) for i in feature_shift]) == 0:
                    continue

                hscale = scale // 2
                if '#' in shape:
                    effective_shift = tuple(0 for _ in feature_shift)
                    negative_extent = tuple(
                        (hscale if s == 0 else hscale * abs(max(0, s)))
                        for d, s in zip(range(ndim), feature_shift)
                    )
                    positive_extent = tuple(
                        (hscale if s == 0 else hscale * abs(min(0, s)))
                        for d, s in zip(range(ndim), feature_shift)
                    )
                elif '|' in shape:
                    effective_shift = tuple(i * hscale for i in feature_shift)
                    negative_extent = tuple(
                        (hscale if s == 0 else abs(max(0, s)))
                        for d, s in zip(range(ndim), feature_shift)
                    )
                    positive_extent = tuple(
                        (hscale if s == 0 else abs(min(0, s)))
                        for d, s in zip(range(ndim), feature_shift)
                    )
                elif '*' in shape:
                    effective_shift = tuple(i * scale for i in feature_shift)
                    negative_extent = tuple(
                        (
                            max(0, hscale * (2 + radius))
                            if s == 0
                            else (abs(s) * scale - 2 if s > 0 else hscale)
                        )
                        for d, s in zip(range(ndim), feature_shift)
                    )
                    positive_extent = tuple(
                        (
                            max(0, hscale * (2 + radius))
                            if s == 0
                            else (abs(s) * scale - 2 if s < 0 else hscale)
                        )
                        for d, s in zip(range(ndim), feature_shift)
                    )
                else:
                    effective_shift = tuple(i * scale for i in feature_shift)
                    negative_extent = (hscale,) * ndim
                    positive_extent = (hscale,) * ndim

                feature_description = (
                    effective_shift,
                    negative_extent,
                    positive_extent,
                    shape,
                )

                # Now we check if the feature overlaps with any excluded voxels:
                # if check_for_excluded_voxels(
                #         effective_shift, negative_extent, positive_extent, excluded_voxels
                # ):
                #     continue

                #  We append the feature description:
                feature_description_list.append(feature_description)

        # Some features might be identical due to the aspect ratio, we eliminate duplicates:
        no_duplicate_feature_description_list = _remove_duplicates(
            feature_description_list
        )
        # We save the last computed feature description list for debug purposes:
        self.debug_feature_description_list = feature_description_list
        # We check and report how many duplicates were eliminated:
        number_of_duplicates = len(feature_description_list) - len(
            no_duplicate_feature_description_list
        )

        lprint(f"Number of duplicate features: {number_of_duplicates}")
        feature_description_list = no_duplicate_feature_description_list
        return feature_description_list

    @property
    def receptive_field_radius(self) -> int:
        radius = 0
        counter = 0
        for width, scale in zip(self.kernel_widths, self.kernel_scales):

            if counter > self.max_level:
                break

            radius = max(radius, width * scale // 2)
            counter += 1

        return radius

    def num_features(self, ndim: int) -> int:
        self._ensure_feature_description_available(ndim)
        return len(self._feature_descriptions_list)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        # Save original image dtype:
        self.original_dtype = image.dtype

        # Scipy does not support float16 yet:
        dtype = (
            numpy.float32
            if self.original_dtype == numpy.float16
            else self.original_dtype
        )
        image = image.astype(dtype=dtype, copy=False)

        self.image = image
        self.excluded_voxels = excluded_voxels
        self.kwargs = kwargs

        # Let's make sure we have the descriptions of the features:
        self._ensure_feature_description_available(image.ndim)

        size_to_feature = {}
        for feature_description in self._feature_descriptions_list:
            # Unpacking the description:
            translation, negative_extent, positive_extent, shape = feature_description

            # Calculating the uniform feature size:
            size = tuple((n + 1 + s for n, s in zip(negative_extent, positive_extent)))

            # Let's check that the feature is not already computed:
            if size not in size_to_feature:
                lprint(f"Pre-computing uniform filter of size: {size}")
                # Compute the feature
                feature = self._compute_uniform_filter(image, size=size)

                # save feature in cache:
                size_to_feature[size] = feature

        self._size_to_full_feature = size_to_feature

    def compute_feature(self, index: int, feature):

        feature_description = self._feature_descriptions_list[index]
        lprint(
            f"Uniform feature: {index}, description: {feature_description}, excluded_voxels={self.excluded_voxels}"
        )

        # Unpacking the description:
        translation, negative_extent, positive_extent, shape = feature_description

        # Calculating the uniform feature size:
        size = tuple((n + 1 + s for n, s in zip(negative_extent, positive_extent)))

        # Fetching the corresponding full feature:
        full_feature = self._size_to_full_feature[size]

        # We use the full uniform filter result and modify it accordingly:
        self._translate_and_exclude_center_value(
            self.image, full_feature, feature, feature_description, self.excluded_voxels
        )

    def _translate_and_exclude_center_value(
        self, image, feature_in, feature_out, feature_description, excluded_voxels
    ):
        """
        It is not recommended to optimise this function as there is some very technical points happening here,
        and the key functions that need optimizing have been externalised anyway (see below).
        """

        # This function exists to facilitate the implementation of optimised versions of it.

        # Unpacking the description:
        translation, negative_extent, positive_extent, shape = feature_description

        # Adjust translation given the extents:
        # There is something a bit tricky here, this term: ' + (p - n) '
        # That's needed to take into account the fact that the center of a
        # uniform filter of even shape does not land on a voxel...
        translation_adjusted = tuple(
            (
                -t + (p - n) // 2
                for t, n, p in zip(translation, negative_extent, positive_extent)
            )
        )

        # We check that the translation is not trivial:
        if any(abs(t) > 0 for t in translation_adjusted):
            self._translate_image(feature_in, feature_out, translation_adjusted)
        else:
            # If the translation does not translated anything then let's just not translate, right?
            feature_out[...] = feature_in

        # We store here the sum of excluded values to be able to substract:
        excluded_values_sum = None
        # And count how many voxels are effectively excluded:
        excluded_count = 0

        for excluded_voxel in excluded_voxels:
            # Is the center voxel within the filter footprint?
            center_value_within_footprint = all(
                -n <= e - t <= p
                for e, t, n, p in zip(
                    excluded_voxel, translation, negative_extent, positive_extent
                )
            )
            if center_value_within_footprint:
                lprint(f"excluded voxel: {excluded_voxel}")

                # We increment the exclusion count:
                excluded_count += 1

                # Just-in-time allocation:
                if excluded_values_sum is None:
                    excluded_values_sum = numpy.zeros_like(image)

                # In this case we remove the excluded value, first we:
                if any(abs(ev) > 0 for ev in excluded_voxel):
                    excluded_values_sum = self._translate_and_add_image(
                        excluded_values_sum, excluded_voxel, image
                    )
                else:
                    _fast_inplace_add(excluded_values_sum, image)

        # We check if we need to touch the feature at all:
        if excluded_count > 0:

            # Calculating the uniform feature size:
            size = tuple((n + 1 + s for n, s in zip(negative_extent, positive_extent)))

            # we compute the volume of the footprint in voxels:
            footprint_volume = prod(size)

            # If we have more than one excluded voxel, we need to be carefull: if the proportion of excluded voxels
            # in a feature becomes too large (> 10%), we need  to exclude that feature entirely, otherwise it will be
            # too much difference between the feature with or without excluded voxels, and thus too confusing for the regressor.
            # Some of this comes from empirical evidence.
            num_of_excluded_voxels = len(excluded_voxels)
            exclude_feature_entirely = num_of_excluded_voxels > 1 and (
                excluded_count / footprint_volume > 0.1
            )

            if exclude_feature_entirely:
                # If all voxels of the footprint are excluded then the whole feature must be zeroed out:
                feature_out[...] = 0
            else:
                # Then we compute the correction factor so that the feature is the average of the remaining voxels:
                _apply_correction(
                    feature_out, excluded_values_sum, footprint_volume, excluded_count
                )

    def _compute_uniform_filter(self, image, size):
        """
        Override this method to provide an accelerated version
        """

        def no_cuda_cpu_mode():
            # No CUDA? we use CPU mode instead:
            # Different methods perform differently based on filter size:
            max_size = max(size) if isinstance(size, tuple) else size
            if max_size > 128:
                # Numba scales well for large filter sizes:
                output = numba_cpu_uniform_filter(image, size=size, mode="nearest")
                lprint(f"Computed filter of size: {size} with Numba")
            else:
                # Scipy parallel is more efficient for small filter sizes:
                output = parallel_uniform_filter(image, size=size, mode="nearest")
                lprint(f"Computed filter of size: {size} with parallel scipy")

            return output

        if image.size < 1024:
            lprint("Image too small, CUDA not needed!")
            output = no_cuda_cpu_mode()
        else:
            try:
                # No point of using CUDA for very small images!
                # Let's try CUDA first:
                # Note: this is not optimal as the image is pushed to GPU every time...
                from aydin.util.fast_uniform_filter.numba_gpu_uf import (
                    numba_gpu_uniform_filter,
                )

                if self.cuda_stream is None:
                    self.cuda_stream = cuda.stream()

                if self.gpu_image is None:
                    image = numpy.ascontiguousarray(image)
                    self.gpu_image = cuda.to_device(image, stream=self.cuda_stream)

                output = numba_gpu_uniform_filter(
                    self.gpu_image,
                    size=size,
                    mode="nearest",
                    cuda_stream=self.cuda_stream,
                )
                lprint(f"Computed filter of size: {size} with CUDA")
            except Exception as e:
                if isinstance(e, CudaSupportError):
                    lprint(
                        "CUDA not supported on this machine, falling back to numba and scipy."
                    )
                else:
                    import sys

                    error_str = (str(sys.exc_info()[0]) + ', and: ' + str(e)).replace(
                        '\n', ', '
                    )
                    lprint(
                        f"Cannot use CUDA for computing uniform filter because of: {error_str}"
                    )
                output = no_cuda_cpu_mode()

        # Ensure correct type:
        dtype = image.dtype if self.dtype is None else self.dtype
        dtype = numpy.float32 if dtype == numpy.float16 else dtype
        output = output.astype(dtype=dtype, copy=False)

        return output

    def _translate_and_add_image(self, sum_image, translation, image):
        """
        Override this method to provide an accelerated version
        """
        fast_shift(image, shift=tuple(translation), output=sum_image, add=True)
        return sum_image

    def _translate_image(self, feature_in, feature_out, translation):
        """
        Override this method to provide an accelerated version
        """
        fast_shift(feature_in, shift=tuple(translation), output=feature_out)

    def finish(self):
        # Here we cleanup any resource alocated for the last feature computation:
        self.image = None
        self.excluded_voxels = None
        self.kwargs = None
        self._feature_descriptions_list = None
        self._size_to_full_feature = None
        self.gpu_image = None
        self.cuda_stream = None


def _apply_correction(
    feature, excluded_values_sum, footprint_volume: int, excluded_count: int
):
    if (
        feature.dtype == numpy.int16
        or feature.dtype == numpy.uint16
        or feature.dtype == numpy.int8
        or feature.dtype == numpy.uint8
    ):
        # feature_float = feature.astype(dtype=numpy.float32, copy=False)
        # _apply_correction_numba(feature_float, excluded_values_sum, footprint_volume, excluded_count)
        # feature[...] = feature_float.astype(dtype=feature.dtype, copy=False)
        alpha = footprint_volume / (footprint_volume - excluded_count)
        beta = -alpha / footprint_volume  # noqa: F841

        numexpr.evaluate(
            "alpha*feature + beta*excluded_values_sum", out=feature, casting='unsafe'
        )
    else:
        _apply_correction_numba(
            feature, excluded_values_sum, footprint_volume, excluded_count
        )


@jit(
    nopython=True,
    # parallel=True,
    error_model='numpy',
    fastmath={'contract', 'afn', 'reassoc'},
)
def _apply_correction_numba(
    feature, excluded_values_sum, footprint_volume: int, excluded_count: int
):
    alpha = footprint_volume / (footprint_volume - excluded_count)
    beta = -alpha / footprint_volume

    feature *= alpha
    feature += beta * excluded_values_sum


@jit(
    nopython=True,
    parallel=True,
    error_model='numpy',
    fastmath={'contract', 'afn', 'reassoc'},
)
def _fast_inplace_add(a, b):
    a += b


def prod(atuple: Tuple[Union[float, int]]):
    # In python 3.8 there is a prod function in math, until then we have:
    return reduce(mul, atuple)
