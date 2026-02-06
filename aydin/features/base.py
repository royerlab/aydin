"""Base classes for feature generation in the Aydin denoising framework."""

import os
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Optional, Tuple

import jsonpickle
import numpy
from numpy import ndarray

from aydin.util.log.log import lprint, lsection
from aydin.util.misc.json import encode_indent
from aydin.util.offcore.offcore import offcore_array


class FeatureGeneratorBase(ABC):
    """Abstract base class for all feature generators.

    Feature generators transform input images into multi-dimensional feature
    arrays that can be used by regression models for self-supervised denoising.
    Each feature captures different aspects of the local image structure at
    various scales.

    Attributes
    ----------
    check_nans : bool
        When True, enables NaN checking in feature arrays for debugging.
    debug_force_memmap : bool
        When True, forces memory-mapped arrays for feature storage.
    dtype : numpy.dtype or None
        Data type for feature arrays. If None, the input image dtype is used.
    """

    _max_non_batch_dims = 4
    _max_voxels = 512**3

    def __init__(self):
        """Construct a feature generator with default settings."""

        self.check_nans = False
        self.debug_force_memmap = False

        # Implementations must initialise the dtype so that feature arrays can be created with correct type:
        self.dtype = None

    def save(self, path: str):
        """Save the feature generator to a folder as a JSON file.

        Serializes the complete state of this feature generator using
        jsonpickle, writing it to ``feature_generation.json`` inside the
        specified directory.

        Parameters
        ----------
        path : str
            Directory path where the feature generator will be saved.
            The directory is created if it does not exist.

        Returns
        -------
        frozen : str
            The JSON-serialized representation of this feature generator.
        """

        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving feature generator to: {path}")
        with open(join(path, "feature_generation.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """Load a feature generator from a folder.

        Deserializes a feature generator from a ``feature_generation.json``
        file inside the specified directory.

        Parameters
        ----------
        path : str
            Directory path from which to load the feature generator.

        Returns
        -------
        thawed : FeatureGeneratorBase
            The deserialized feature generator instance.
        """

        lprint(f"Loading feature generator from: {path}")
        with open(join(path, "feature_generation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed._load_internals(path)

        return thawed

    @abstractmethod
    def _load_internals(self, path: str):
        """Load any internal state not captured by JSON serialization.

        Parameters
        ----------
        path : str
            Directory path from which to load internal state.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_receptive_field_radius(self):
        """Return the receptive field radius in pixels.

        The receptive field radius is the maximum distance (in pixels) from
        the center voxel that any feature in this generator can reach.

        Returns
        -------
        radius : int
            Receptive field radius in pixels.
        """
        raise NotImplementedError()

    @abstractmethod
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
        Compute features for the given image.

        If the input image is of shape ``(b, c, *spatial_dims)`` (batch,
        channel, spatial dims), the resulting features are of shape
        ``(b, *spatial_dims, n)`` when ``feature_last_dim=True``, where ``n``
        is the total number of features.

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
        raise NotImplementedError()

    def create_feature_array(self, image, nb_features):
        """Create a feature array of the appropriate size and dtype.

        May use memory mapping for large arrays to avoid exceeding available
        RAM.

        Parameters
        ----------
        image : numpy.ndarray
            Image for which features are being created. Used to determine
            the spatial dimensions and default dtype.
        nb_features : int
            Number of features (size of the feature dimension).

        Returns
        -------
        array : numpy.ndarray
            Feature array of shape ``(nb_features, batch_size, *spatial_dims)``
            with the appropriate dtype.
        """

        with lsection(f'Creating feature array for image of shape: {image.shape}'):
            # That's the shape we need:
            shape = (nb_features, image.shape[0]) + image.shape[2:]
            dtype = image.dtype if self.dtype is None else self.dtype
            dtype = numpy.float32 if dtype == numpy.float16 else dtype
            array = offcore_array(
                shape=shape, dtype=dtype, force_memmap=self.debug_force_memmap
            )
            return array
