from typing import Optional
import numpy

from aydin.it.base import ImageTranslatorBase
from aydin.it.transforms.padding import PaddingTransform
from aydin.util.log.log import lsection


class ImageTranslatorDeconvBase(ImageTranslatorBase):
    """Base class for Image Deconvolution"""

    def __init__(
        self,
        psf_kernel,
        *args,
        clip: bool = True,
        padding: Optional[int] = None,
        padding_mode=None,
        dtype=numpy.float32,
        max_voxels_per_tile: int = None,
        **kwargs,
    ):
        """Constructs a base deconvolution object

        Parameters
        ----------
        psf_kernel : numpy.typing.ArrayLike
            2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        args
        clip : bool
        padding : int, optional
        padding_mode
        dtype
        max_voxels_per_tile : int
        kwargs
        """
        super().__init__(*args, **kwargs)

        self.psf_kernel_numpy = psf_kernel
        self.clip = clip
        self.dtype = dtype
        self.__debug_allocation = False

        # default padding mode:
        if padding_mode is None:
            padding_mode = 'reflect'

        # remove existing padding:
        self.transforms_list = list(
            [t for t in self.transforms_list if not isinstance(t, PaddingTransform)]
        )
        padding = (
            max(max(7, s) for s in psf_kernel.shape) if padding is None else padding
        )
        # add correct padding:
        self.transforms_list.append(
            PaddingTransform(pad_width=padding, mode=padding_mode)
        )

        self.max_voxels_per_tile = max_voxels_per_tile

        self.tile_min_margin = max(psf_kernel.shape)

    def save(self, path: str):
        """Saves a 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------
        frozen

        """
        with lsection(f"Saving Lucy-Richardson image translator to {path}"):
            frozen = super().save(path)

        return frozen

    def _load_internals(self, path: str):
        """
        Method to load internals.

        Parameters
        ----------
        path : str
            path to load from

        """
        with lsection(f"Loading Lucy-Richardson image translator from {path}"):
            # no internals to load here...
            pass

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # nothing to do here...
        return state

    def stop_training(self):
        """Stop training"""
        pass
        # we can't do that... for now...

    def _estimate_memory_needed_and_available(self, image):
        """
        Method to estimate needed and available memory amounts.

        Parameters
        ----------
        image : array_like

        Returns
        -------
        Tuple of memory_needed, memory_available

        """
        # By default there is no memory needed which means no constraints

        memory_needed, memory_available = super()._estimate_memory_needed_and_available(
            image
        )
        # TODO: this is a rough estimate, it is not clear how much is really needed...
        memory_needed = 6 * image.size * image.dtype.itemsize

        return memory_needed, memory_available

    def _debug_allocation(self, info):
        """
        Method to log used memory by CUDA with cupy backend.

        Parameters
        ----------
        info

        """
        pass
