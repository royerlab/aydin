"""Integral stack deskewing transform.

Provides a transform that applies integral (non-interpolating) shear
transformations to correct stack skew, bringing spatially correlated
voxels closer together to improve denoising performance.
"""

import numpy
from numpy.typing import ArrayLike

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import aprint, asection


class DeskewTransform(ImageTransformBase):
    """Integral stack deskewer.

    Denoising is more effective if voxels carrying correlated signal are
    close to each other. When a stack is skewed -- as resulting in some
    imaging modalities -- correlated voxels that should be close in space
    are far from each other. Thus, deskewing the image before denoising is
    highly recommended. Importantly, the deskewing must be 'integral',
    meaning that it must not interpolate voxel values, which is an
    inadvisable lossy operation. Integral stack deskewing consists of
    applying an integral shear transformation to a stack. Two axes need to
    be specified: the 'z'-axis and the 'skew'-axis along which shifting
    happens. The delta parameter controls the amount of shift per plane and
    must be an integer. We automatically snap the delta value to the closest
    integer. Padding is supported.

    Note: this only works for images with at least 3 dimensions. Does
    nothing on images with less than 3 dimensions. (advanced)
    <notgui>
    """

    preprocess_description = "Deskew image" + ImageTransformBase.preprocess_description
    postprocess_description = (
        "Reskew image" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        delta: float = 0,
        z_axis: int = 0,
        skew_axis: int = 1,
        pad: bool = True,
        priority: float = 0.4,
        **kwargs,
    ):
        """Construct a DeskewTransform.

        Parameters
        ----------
        delta : float
            How much shifting from one plane to the next
        z_axis : int
            Axis for which the amount of shift depends upon.
        skew_axis : int
            Axis over which the image is shifted.
        pad : bool
            True for padding before rolling, this is useful
            because normal padding is rarely enough.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocessing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.delta = int(round(delta))
        self.z_axis = z_axis
        self.skew_axis = skew_axis
        self.pad = pad

        aprint(f"Instantiating: {self}")

    def __getstate__(self):
        """Return picklable state (no fields excluded for this transform).

        Returns
        -------
        dict
            Complete object state dictionary.
        """
        state = self.__dict__.copy()
        # nothing to exclude
        return state

    def __str__(self):
        """Return a human-readable string representation.

        Returns
        -------
        str
            String showing the class name and key parameters.
        """
        return (
            f'{type(self).__name__} (delta={self.delta},'
            f' z_axis={self.z_axis},'
            f' skew_axis={self.skew_axis},'
            f' pad={self.pad} )'
        )

    def __repr__(self):
        """Return a detailed string representation.

        Returns
        -------
        str
            Same as ``__str__``.
        """
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        """Deskew the image stack by applying an integral shear transform.

        Only acts on images with 3 or more dimensions.

        Parameters
        ----------
        array : ArrayLike
            Input image array.

        Returns
        -------
        numpy.ndarray
            Deskewed image array.
        """
        with asection(
            f"Deskewing (delta={self.delta}, z_axis={self.z_axis}, skew_axis={self.skew_axis}, pad={self.pad}) array of shape: {array.shape} and dtype: {array.dtype}:"
        ):
            if array.ndim >= 3:
                return self.deskew(array)
            else:
                return array

    def postprocess(self, array: ArrayLike):
        """Re-skew the image to undo the deskewing applied during preprocessing.

        Only acts on images with 3 or more dimensions.

        Parameters
        ----------
        array : ArrayLike
            Denoised image array.

        Returns
        -------
        numpy.ndarray
            Re-skewed image array.
        """
        if not self.do_postprocess:
            return array
        with asection(
            f"Undoing deskew for array of shape: {array.shape} and dtype: {array.dtype}:"
        ):
            if array.ndim >= 3:
                return self.reskew(array)
            else:
                return array

    def deskew(self, array: ArrayLike, pad_mode='wrap'):
        """Apply the deskew (forward shear) transformation.

        Parameters
        ----------
        array : ArrayLike
            Input array to deskew.
        pad_mode : str
            Padding mode for numpy.pad (default: 'wrap').

        Returns
        -------
        numpy.ndarray
            Deskewed array.
        """
        array = self._permutate(array)
        array = self._skew_transform(
            array, self.delta, pad=True, crop=False, pad_mode=pad_mode
        )
        array = self._depermutate(array)
        return array

    def reskew(self, array: ArrayLike):
        """Apply the reskew (inverse shear) transformation.

        Parameters
        ----------
        array : ArrayLike
            Array to reskew.

        Returns
        -------
        numpy.ndarray
            Reskewed array.
        """
        array = self._permutate(array)
        array = self._skew_transform(
            array, -self.delta, pad=False, crop=True, pad_mode=''
        )
        array = self._depermutate(array)
        return array

    def _permutate(self, array: ArrayLike):
        """Permute axes so z and skew axes come first.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        numpy.ndarray
            Array with z_axis and skew_axis moved to positions 0 and 1.
        """
        permutation = self._get_permutation(array)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _depermutate(self, array: ArrayLike):
        """Inverse permutation to restore original axis ordering.

        Parameters
        ----------
        array : ArrayLike
            Permuted array.

        Returns
        -------
        numpy.ndarray
            Array with original axis ordering restored.
        """
        permutation = self._get_permutation(array, inverse=True)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _get_permutation(self, array: ArrayLike, inverse=False):
        """Compute the axis permutation for deskewing.

        Parameters
        ----------
        array : ArrayLike
            Array whose dimensions define the permutation.
        inverse : bool
            If True, return the inverse permutation.

        Returns
        -------
        tuple of int or numpy.ndarray
            Axis permutation (or its inverse).
        """
        permutation = (self.z_axis, self.skew_axis) + tuple(
            axis
            for axis in range(array.ndim)
            if axis not in [self.z_axis, self.skew_axis]
        )
        if inverse:
            permutation = numpy.argsort(permutation)
        return permutation

    @staticmethod
    def _skew_transform(array: ArrayLike, delta, pad, crop, pad_mode='wrap'):
        """Apply an integral shear transformation along axis 1 indexed by axis 0.

        Assumes that the first dimension (index=0) is the z dimension and
        the second dimension (index=1) is the 'skewed' dimension. The array
        can have arbitrary dimensions after that.

        Parameters
        ----------
        array : ArrayLike
            Input array with z-axis at position 0 and skew-axis at position 1.
        delta : int
            Shift per z-plane (positive or negative).
        pad : bool
            If True, pad the skew axis before shifting.
        crop : bool
            If True, crop the skew axis after shifting to remove padding.
        pad_mode : str
            Padding mode for ``numpy.pad`` (default: 'wrap').

        Returns
        -------
        numpy.ndarray
            Sheared (or un-sheared) array.
        """

        num_z_planes = array.shape[0]
        pad_length = abs(delta * num_z_planes)

        if pad:
            padding = (pad_length, 0) if delta < 0 else (0, pad_length)
            array = numpy.pad(
                array,
                pad_width=((0, 0), padding) + ((0, 0),) * (array.ndim - 2),
                mode=pad_mode,
            )
        else:
            array = array.copy()

        for zi in range(num_z_planes):
            array[zi, ...] = numpy.roll(array[zi, ...], shift=delta * zi, axis=0)

        if crop:
            cropping = (
                slice(pad_length, None, 1) if delta > 0 else slice(0, -pad_length, 1)
            )
            crop_slice = (slice(None), cropping) + (slice(None),) * (array.ndim - 2)
            array = array[crop_slice]

        return array
