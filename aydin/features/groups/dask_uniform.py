import multiprocessing

import dask
import numpy
from dask.distributed import Client
from dask_image.ndfilters import uniform_filter

from aydin.features.groups.uniform import UniformFeatures

n_workers = max(1, int(0.9 * multiprocessing.cpu_count()))
_client = Client(n_workers=n_workers, set_as_default=False)


class DaskUniformFeatures(UniformFeatures):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def prepare(self, image, excluded_voxels=None, **kwargs):

        # Save original image dtype:
        self.original_dtype = image.dtype

        # Numba does not support float16 yet:
        dtype = (
            numpy.float32
            if self.original_dtype == numpy.float16
            else self.original_dtype
        )
        image = image.astype(dtype=dtype, copy=False)

        super().prepare(image, excluded_voxels=excluded_voxels, kwargs=kwargs)

        if image.ndim == 1:
            chunk_size = 1024 * 1024
        elif image.ndim == 2:
            chunk_size = 1024
        elif image.ndim == 3:
            chunk_size = 128
        elif image.ndim == 3:
            chunk_size = 32

        chunks = (min(s, chunk_size) for s in image.shape)
        self.dask_image = dask.from_array(image, chunks=chunks)

    def _compute_uniform_filter(self, image, size: int = 3):
        """
        Overriding this method from parent class to implement accelerated version

        Parameters
        ----------
        image : numpy.ndarray
        size : int

        Returns
        -------
        output : numpy.ndarray

        """

        output = uniform_filter(image, size=size, mode="nearest")

        output = _client.submit(output)

        # Make sure that output image has correct dtype:
        output = output.astype(dtype=self.original_dtype, copy=False)

        return output
