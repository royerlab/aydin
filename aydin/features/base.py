import os
from abc import ABC, abstractmethod
from os.path import join
import jsonpickle
import numpy

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
        Saves a 'all-batteries-inlcuded' feature generator at a given path (folder)

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
        exclude_center_feature=False,
        exclude_center_value=False,
        features=None,
        feature_last_dim=True,
        passthrough_channels=None,
        num_reserved_features=0,
        excluded_voxels=None,
        spatial_feature_offset=None,
        spatial_feature_scale=None,
    ):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (n,d,h,w) where n is the number of features.

        Parameters
        ----------
        image : numpy.ndarray
            image for which features are computed
        exclude_center_feature : bool
        exclude_center_value : bool
        features
        feature_last_dim : bool
        passthrough_channels
        num_reserved_features : int
        excluded_voxels
        spatial_feature_offset
        spatial_feature_scale

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
