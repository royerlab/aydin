import numpy

from numpy.typing import ArrayLike

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class DeskewTransform(ImageTransformBase):
    """(Integral) Stack Deskewer

    Denoising is more effective if voxels carrying correlated signal are close to each other. When a stack is skewed
    -- as resulting in some imaging modalities -- correlated voxels that should be close in space are far from each
    other. Thus, deskewing the image before denoising is highly recommended. Importantly, the deskewing must be
    'integral', meaning that it must not interpolate voxel values, which is a unadvised lossy operation. Integral
    stack deskewing consists in applying an integral shear transformation to a stack. Two axes need to be specified:
    the 'z'-axis and the 'skew'-axis along which shifting happens. The delta parameter controls the amount of shift
    per plane - must be an integer. We automatically snap the delta value to the closest integer. Padding is supported.

    Note: this only works for images with at least 3 dimensions. Does nothing
    on images with less than 3 dimensions.(advanced)
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
        """
        Constructs a stack deskewer

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
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.delta = int(round(delta))
        self.z_axis = z_axis
        self.skew_axis = skew_axis
        self.pad = pad

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # nothing to exclude
        return state

    def __str__(self):
        return (
            f'{type(self).__name__} (delta={self.delta},'
            f' z_axis={self.z_axis},'
            f' skew_axis={self.skew_axis},'
            f' pad={self.pad} )'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        with lsection(
            f"Deskewing (delta={self.delta}, z_axis={self.z_axis}, skew_axis={self.skew_axis}, pad={self.pad}) array of shape: {array.shape} and dtype: {array.dtype}:"
        ):
            if array.ndim >= 3:
                return self.deskew(array)
            else:
                return array

    def postprocess(self, array: ArrayLike):
        if not self.do_postprocess:
            return array
        with lsection(
            f"Undoing deskew for array of shape: {array.shape} and dtype: {array.dtype}:"
        ):
            if array.ndim >= 3:
                return self.reskew(array)
            else:
                return array

    def deskew(self, array: ArrayLike, pad_mode='wrap'):
        array = self._permutate(array)
        array = self._skew_transform(
            array, self.delta, pad=True, crop=False, pad_mode=pad_mode
        )
        array = self._depermutate(array)
        return array

    def reskew(self, array: ArrayLike):
        array = self._permutate(array)
        array = self._skew_transform(
            array, -self.delta, pad=False, crop=True, pad_mode=''
        )
        array = self._depermutate(array)
        return array

    def _permutate(self, array: ArrayLike):
        permutation = self._get_permutation(array)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _depermutate(self, array: ArrayLike):
        permutation = self._get_permutation(array, inverse=True)
        array = numpy.transpose(array, axes=permutation)
        return array

    def _get_permutation(self, array: ArrayLike, inverse=False):
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
        """
        This method assumes that the first dimension (index=0) is the z dimension,
        and the second dimension (index=1) is the 'skewed' dimension.
        The array can have arbitrary dimensions after that...
        We also assume that the array has been properly padded so that we can 'roll' the
        skewed dimension without fear or regret.
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
