"""Super-fast representative crop extraction via downscaled search.

Uses image downscaling to accelerate the representative crop search,
then maps the result back to the original image resolution.
"""

from memoization.memoization import cached
from numpy.typing import ArrayLike
from scipy.ndimage import zoom

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import aprint, asection


@cached(ttl=10, max_size=5)
def super_fast_representative_crop(
    image: ArrayLike,
    crop_size: int,
    min_length: int = 8,
    search_mode: str = 'systematic',
    granularity_factor: int = 3,
    return_slice: bool = False,
    min_scaling_factor: int = 2,
    *args,
    **kwargs,
):
    """Extract a representative crop from an image using downscaled search.

    Downscales the image first for faster crop search, then maps the
    best crop location back to the original resolution. Results are
    cached with a 10-second TTL.

    Parameters
    ----------
    image : ArrayLike
        Image to extract representative crop from.
    crop_size : int
        Desired crop size in voxels.
    min_length : int, optional
        Minimum allowed axis length for the crop, by default 8.
    search_mode : str, optional
        Search mode for best crops: ``'random'`` or ``'systematic'``.
        By default ``'systematic'``.
    granularity_factor : int, optional
        Granularity of search. Higher values correspond to more overlap
        between candidate crops, by default 3.
    return_slice : bool, optional
        If True, returns a tuple of (crop, slice_tuple), by default False.
    min_scaling_factor : int, optional
        Minimum downscaling factor per axis, by default 2.
    *args
        Additional positional arguments passed to ``representative_crop``.
    **kwargs
        Additional keyword arguments passed to ``representative_crop``.

    Returns
    -------
    numpy.ndarray or tuple
        The most representative crop. If ``return_slice`` is True,
        returns a tuple of (crop_array, slice_tuple).
    """
    with asection(f"Super fast cropping image of size: {image.shape}"):

        # Compute downscale facto per dimension:
        def _downscale(length):
            """Compute downscale factor for a given dimension length."""
            return min(max(min_scaling_factor, length // 256), min_length)

        downscale_factor = tuple(
            _downscale(s) if s >= min_length else min_length // 2 for s in image.shape
        )
        aprint(f"Scaling by factors: {downscale_factor}")

        # Compute zoom factor
        zoom_per_axis = tuple(
            1.0 / d if s > d else 1 for d, s in zip(downscale_factor, image.shape)
        )
        aprint(f"zoom_per_axis: {zoom_per_axis}")

        # Downsample image:
        with asection(f"Downscaling image of shape: {image.shape}..."):
            image_d = zoom(image, zoom=zoom_per_axis, prefilter=False, order=0)

        # Compute overall zoom factor:
        overall_zoom = image_d.size / image.size

        # Compute the scaled-down crop_size:
        crop_size = int(crop_size * overall_zoom)

        # Delegate cropping:
        _, slice_ = representative_crop(
            image_d,
            crop_size=crop_size,
            search_mode=search_mode,
            granularity_factor=granularity_factor,
            min_length=min_length,
            return_slice=True,
            *args,
            **kwargs,
        )

        # Normalise Slice:
        # Upscale slice:
        slice_ = tuple(
            slice(
                0 if sl.start is None else sl.start,
                s if sl.stop is None else sl.stop,
                1,
            )
            for sl, s in zip(slice_, image_d.shape)
        )

        # Upscale slice:
        slice_ = tuple(
            slice(sl.start * s, sl.stop * s, 1)
            for sl, s in zip(slice_, downscale_factor)
        )

        # Clip slice to dimensions of image:
        slice_ = tuple(
            slice(max(sl.start, 0), min(sl.stop, s), 1)
            for sl, s in zip(slice_, image.shape)
        )

        # Crop Image:
        crop = image[slice_]

        # Returns:
        if return_slice:
            # Return slice if requested:
            return crop, slice_
        else:
            return crop
