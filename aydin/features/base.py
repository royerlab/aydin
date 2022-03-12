import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Optional, Tuple, List

import jsonpickle
import numpy
from numpy import ndarray

from aydin.util.misc.json import encode_indent
from aydin.util.log.log import lprint, lsection
from aydin.util.offcore.offcore import offcore_array


class FeatureGeneratorBase(ABC):
    """
    Feature Generator base class
    """

    _max_non_batch_dims = 4
    _max_voxels = 512 ** 3

    def __init__(self):
        """
        Constructs a feature generator
        """

        self.check_nans = False
        self.debug_force_memmap = False

        # Implementations must initialise the dtype so that feature arrays can be created with correct type:
        self.dtype = None

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' feature generator at a given path (folder)

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------
        frozen

        """

        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving feature generator to: {path}")
        with open(join(path, "feature_generation.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """
        Returns a 'all-batteries-inlcuded' feature generator from a given path (folder)

        Parameters
        ----------
        path : str
            path to load from

        Returns
        -------
        thawed

        """

        lprint(f"Loading feature generator from: {path}")
        with open(join(path, "feature_generation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed._load_internals(path)

        return thawed

    @abstractmethod
    def _load_internals(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def get_receptive_field_radius(self):
        """
        Returns the receptive field radius in pixels
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
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (n,d,h,w) where n is the number of features.

        Parameters
        ----------
        image : numpy.ndarray
            image for which features are computed

        exclude_center_feature : bool
            If true, features that use the image
            patch's center pixel are entirely excluded from teh set of computed
            features.

        exclude_center_value : bool
            If true, the center pixel is never used
            to compute any feature, different feature generation algorithms can
            take different approaches to acheive that.

        features : ndarray
            If None the feature array is allocated internally,
            if not None the provided array is used to store the features.

        feature_last_dim : bool
            If True the last dimension of the feature
            array is the feature dimension, if False then it is the first
            dimension.

        passthrough_channels : Optional[Tuple[bool]]
            Optional tuple of booleans that specify which channels are 'pass-through'
            channels, i.e. channels that are not featurised and directly used as features.

        num_reserved_features : int
            Number of features to be left as blank,
            useful when adding features separately.

        excluded_voxels : Optional[List[Tuple[int]]]
            List of pixel coordinates -- expressed as tuple of ints relative to the central pixel --
            that will be excluded from any computed features. This is used for implementing
            'extended blind-spot' N2S denoising approaches.

        spatial_feature_offset: Optional[Tuple[float, ...]]
            Offset vector to be applied (added) to the spatial features (if used).

        spatial_feature_scale: Optional[Tuple[float, ...]]
            Scale vector to be applied (multiplied) to the spatial features (if used).

        Returns
        -------
        feature array : numpy.ndarray

        """
        raise NotImplementedError()

    def create_feature_array(self, image, nb_features):
        """
        Creates a feature array of the right size and possibly in a 'lazy' way using memory mapping.

        Parameters
        ----------
        image : numpy.ndarray
            image for which features are created
        nb_features : int

        Returns
        -------
        feature array : numpy.ndarray

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
