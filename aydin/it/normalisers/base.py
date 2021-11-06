import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Tuple
import jsonpickle
import numexpr
import numpy

from aydin.util.misc.json import encode_indent
from aydin.util.log.log import lprint


class NormaliserBase(ABC):
    """Normaliser base class

    Parameters
    ----------
    clip : bool
    epsilon : float
    shape_normalisation : bool
    transform : str

    """

    epsilon: float
    leave_as_float: bool
    clip: bool
    original_dtype: numpy.dtype

    def __init__(self, clip=True, epsilon=1e-21):
        """Constructs a normalisers

        Parameters
        ----------
        clip : bool
        epsilon : float
        """
        self.epsilon = epsilon
        self.clip = clip

        self.rmin = None
        self.rmax = None

        self.axis_permutation = None
        self.permutated_image_shape = None

    def save(self, path: str, name='default'):
        """Saves an 'all-batteries-included' normalisers at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to
        name : str

        Returns
        -------
        json
            frozen
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving normalisers to: {path}")
        with open(join(path, f"normaliser_{name}.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str, name='default'):
        """Returns an 'all-batteries-included' normalisers from a given path (folder).

        Parameters
        ----------
        path : str
            path to load from.
        name : str

        Returns
        -------
        str
            thawed

        """
        lprint(f"Loading normalisers from: {path}")
        with open(join(path, f"normaliser_{name}.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        return thawed

    @abstractmethod
    def calibrate(self, array) -> Tuple[float, float]:
        """Calibrates this normalisers given an array.

        Parameters
        ----------
        array : numpy.ndarray
            array to use for calibration

        Returns
        -------
        array : numpy.ndarray

        """
        raise NotImplementedError()

    def normalise(self, array):
        """Normalises the given array in-place (if possible).

        Parameters
        ----------
        array : numpy.ndarray
            array to normalise
        batch_dims : list
        channel_dims : list

        Returns
        -------
        array : numpy.ndarray

        """
        array = array.astype(numpy.float32, copy=True)

        if self.rmin is not None and self.rmax is not None:
            min_value = numpy.float32(self.rmin)
            max_value = numpy.float32(self.rmax)
            epsilon = numpy.float32(self.epsilon)

            try:
                # We perform operation in-place with numexpr if possible:
                numexpr.evaluate(
                    "((array - min_value) / ( max_value - min_value + epsilon ))",
                    out=array,
                )
                if self.clip:
                    numexpr.evaluate(
                        "where(array<0,0,where(array>1,1,array))", out=array
                    )

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
        """Denormalises the given array in-place (if possible).

        Parameters
        ----------
        array : numpy.ndarray
        denormalise_values : bool
        leave_as_float : bool
        clip : bool

        Returns
        -------
        array : numpy.ndarray

        """

        # we copy the array to preserve the original array:
        array = numpy.copy(array)

        if denormalise_values:
            if self.rmin is not None and self.rmax is not None:

                min_value = numpy.float32(self.rmin)
                max_value = numpy.float32(self.rmax)
                epsilon = numpy.float32(self.epsilon)

                try:
                    # We perform operation in-place with numexpr if possible:

                    if self.clip and clip:
                        numexpr.evaluate(
                            "where(array<0,0,where(array>1,1,array))",
                            out=array,
                            casting='unsafe',
                        )

                    numexpr.evaluate(
                        "array * (max_value - min_value + epsilon) + min_value ",
                        out=array,
                        casting='unsafe',
                    )

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
