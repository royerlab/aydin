from memoization.memoization import cached
from numpy.typing import ArrayLike
from scipy.ndimage import zoom

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import lprint, lsection


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
    """

    Parameters
    ----------
    Same parameters as 'representative_crop' with the addition of:

    min_scaling_factor: int
        Minimal downscaling factor per axis.


    Returns
    -------
    Most representative crop, and if return_slice is True the actual slice object too.

    """
    with lsection(f"Super fast cropping image of size: {image.shape}"):

        # Compute downscale facto per dimension:
        def _downscale(length):
            return min(max(min_scaling_factor, length // 256), min_length)

        downscale_factor = tuple(
            _downscale(s) if s >= min_length else min_length // 2 for s in image.shape
        )
        lprint(f"Scaling by factors: {downscale_factor}")

        # Compute zoom factor
        zoom_per_axis = tuple(1.0 / d if s > 1 else 1 for d, s in zip(downscale_factor, image.shape))
        lprint(f"zoom_per_axis: {zoom_per_axis}")

        # Downsample image:
        with lsection(f"Downscaling image of shape: {image.shape}..."):
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
