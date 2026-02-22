"""Base class for value normalisers.

This module defines `NormaliserBase`, the abstract base class for all value
normalisers that map image intensities to a standard [0, 1] range. Includes
Numba-accelerated normalization and denormalization operations.
"""

import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Tuple

import jsonpickle
import numpy
from numba import jit, prange

from aydin.util.log.log import aprint
from aydin.util.misc.json import encode_indent


class NormaliserBase(ABC):
    """Abstract base class for value normalisers.

    Provides the interface and shared implementation for normalizing image
    values to the [0, 1] range and denormalizing back to the original range
    and dtype. Uses Numba JIT compilation for performance.

    Attributes
    ----------
    epsilon : float
        Small value added to the range denominator to prevent division by zero.
    clip : bool
        If True, clip normalized values to the [0, 1] range.
    rmin : float or None
        Calibrated minimum value for normalization.
    rmax : float or None
        Calibrated maximum value for normalization.
    original_dtype : numpy.dtype
        Original data type of the calibration array, set during ``calibrate()``.
    axis_permutation : list of int or None
        Axis permutation used during shape normalization.
    permutated_image_shape : tuple of int or None
        Intermediate shape used during shape normalization.
    """

    epsilon: float
    clip: bool
    original_dtype: numpy.dtype

    def __init__(self, clip=True, epsilon=1e-21):
        """Construct a NormaliserBase.

        Parameters
        ----------
        clip : bool
            If True, clip normalized values to [0, 1] range.
        epsilon : float
            Small value added to the range denominator to prevent
            division by zero. Default is 1e-21.
        """
        self.epsilon = epsilon
        self.clip = clip

        self.rmin = None
        self.rmax = None

        self.axis_permutation = None
        self.permutated_image_shape = None

    def save(self, path: str, name='default'):
        """Save the normaliser state to a JSON file.

        Parameters
        ----------
        path : str
            Directory path to save to.
        name : str
            Name suffix for the JSON file. Default is 'default'.

        Returns
        -------
        str
            JSON string of the serialized normaliser.
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        aprint(f"Saving normalisers to: {path}")
        with open(join(path, f"normaliser_{name}.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str, name='default'):
        """Load a normaliser from a JSON file.

        Parameters
        ----------
        path : str
            Directory path to load from.
        name : str
            Name suffix for the JSON file. Default is 'default'.

        Returns
        -------
        NormaliserBase
            The restored normaliser instance.
        """
        aprint(f"Loading normalisers from: {path}")
        with open(join(path, f"normaliser_{name}.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        return thawed

    @abstractmethod
    def calibrate(self, array) -> Tuple[float, float]:
        """Calibrate the normaliser from the given array.

        Subclasses must implement this to compute the normalization
        range (rmin, rmax) from the input data.

        Parameters
        ----------
        array : numpy.ndarray
            Array to use for calibration.

        Returns
        -------
        tuple of (float, float)
            The computed (min_value, max_value) range.
        """
        raise NotImplementedError()

    def normalise(self, array):
        """Normalize array values to the [0, 1] range.

        Uses the calibrated min/max range. Optionally clips to [0, 1].

        Parameters
        ----------
        array : numpy.ndarray
            Array to normalise.

        Returns
        -------
        numpy.ndarray
            Normalized array as float32.
        """
        array = array.astype(numpy.float32, copy=True)

        if self.rmin is not None and self.rmax is not None:
            min_value = numpy.float32(self.rmin)
            max_value = numpy.float32(self.rmax)
            epsilon = numpy.float32(self.epsilon)

            try:
                self.normalize_numba(array, min_value, max_value, epsilon)

                if self.clip:
                    array = numpy.where(array < 0, 0, numpy.where(array > 1, 1, array))

            except ValueError:
                array -= min_value
                array /= max_value - min_value + epsilon
                if self.clip:
                    array = numpy.clip(array, 0, 1)  # , out=array

        return array

    def denormalise(
        self,
        array: numpy.ndarray,
        denormalise_values=True,
        leave_as_float=False,
        clip=True,
    ):
        """Denormalize array values back to the original range and dtype.

        Parameters
        ----------
        array : numpy.ndarray
            Normalized array to denormalize.
        denormalise_values : bool
            If True, reverse the value normalization. Default is True.
        leave_as_float : bool
            If True, keep result as float even if original was integer.
            Default is False.
        clip : bool
            If True, clip values to [0, 1] before denormalization.
            Default is True.

        Returns
        -------
        numpy.ndarray
            Denormalized array, cast back to original dtype unless
            leave_as_float is True.
        """

        # we copy the array to preserve the original array:
        array = numpy.copy(array)

        if denormalise_values:
            if self.rmin is not None and self.rmax is not None:

                min_value = numpy.float32(self.rmin)
                max_value = numpy.float32(self.rmax)
                epsilon = numpy.float32(self.epsilon)

                try:
                    if self.clip and clip:
                        array = numpy.where(
                            array < 0, 0, numpy.where(array > 1, 1, array)
                        )

                    self.denormalize_numba(array, min_value, max_value, epsilon)

                except ValueError:
                    if self.clip and clip:
                        array = numpy.clip(array, 0, 1)  # , out=array
                    array *= max_value - min_value + epsilon
                    array += min_value

            if not leave_as_float and self.original_dtype != array.dtype:
                if numpy.issubdtype(self.original_dtype, numpy.integer):
                    # If we cast back to integer, we need to avoid overflows first!
                    type_info = numpy.iinfo(self.original_dtype)

                    if not (self.clip and clip):
                        array = array + (type_info.min - array.min())
                        array = (array * type_info.max) / array.max()

                    array = array.clip(type_info.min, type_info.max, out=array)
                array = array.astype(self.original_dtype)

        return array

    @staticmethod
    @jit(nopython=True, parallel=True, error_model='numpy')
    def normalize_numba(array, min_value, max_value, epsilon):
        """Numba-accelerated in-place normalization to [0, 1] range.

        Parameters
        ----------
        array : numpy.ndarray
            Array to normalize in-place.
        min_value : float
            Minimum value of the normalization range.
        max_value : float
            Maximum value of the normalization range.
        epsilon : float
            Small value to prevent division by zero.
        """
        for _ in prange(numpy.prod(numpy.array(array.shape))):
            array.flat[_] -= min_value
            array.flat[_] /= max_value - min_value + epsilon

    @staticmethod
    @jit(nopython=True, parallel=True, error_model='numpy')
    def denormalize_numba(array, min_value, max_value, epsilon):
        """Numba-accelerated in-place denormalization from [0, 1] range.

        Parameters
        ----------
        array : numpy.ndarray
            Array to denormalize in-place.
        min_value : float
            Minimum value of the original range.
        max_value : float
            Maximum value of the original range.
        epsilon : float
            Small value used during normalization (for consistency).
        """
        for _ in prange(numpy.prod(numpy.array(array.shape))):
            array.flat[_] *= max_value - min_value + epsilon
            array.flat[_] += min_value
