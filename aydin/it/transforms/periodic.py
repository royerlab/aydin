import numpy
import scipy

from numpy.typing import ArrayLike
from scipy.ndimage import minimum_filter
from skimage.feature import peak_local_max

from aydin.it.classic_denoisers.gaussian import calibrate_denoise_gaussian
from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class PeriodicNoiseSuppressionTransform(ImageTransformBase):
    """Periodic Noise Suppression

    Some images have a form of periodic noise that can be seen as strong
    peaks in their power spectral density. Suppressing these peaks before and
    after denoising is often a good idea. This is tricky to use, use with
    care. Works with non-axis aligned periodic patterns.(advanced)
    """

    preprocess_description = (
        "Periodic noise suppression" + ImageTransformBase.preprocess_description
    )
    postprocess_description = (
        "Reaply periodic noise" + ImageTransformBase.postprocess_description
    )
    postprocess_supported = True
    postprocess_recommended = False

    def __init__(
        self,
        mask_radius: float = 0.01,
        threshold: float = 0.1,
        max_num_peaks: int = 32,
        post_processing_is_inverse: bool = False,
        priority: float = 0.30,
        **kwargs,
    ):

        """
        Constructs a Periodic Noise Suppression Transform

        Parameters
        ----------
        mask_radius : float
            Radius for masking peaks in Fourier spectrum. Expressed in units
            relative to the size of the image, i.e. must be within [0,1].
            Will always be at least a single pixel wide.
        threshold : float
            Threshold for FFT peak detection.
        max_num_peaks : int
            Maximum number of peaks in the Fourier spectra to correct for.
        post_processing_is_inverse : bool
            If False, we reapply the periodic suppression to eliminate any
            remaining periodic noise after denoising.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.mask_radius = mask_radius
        self.threshold = threshold
        self.max_num_peaks = max_num_peaks
        self.post_processing_is_inverse = post_processing_is_inverse
        self._original_dtype = None
        self._coordinates = {}

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_original_dtype']
        del state['_coordinates']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (mask_radius={self.mask_radius},'
            f' threshold={self.threshold},'
            f' post_processing_is_inverse={self.post_processing_is_inverse})'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):

        with lsection(
            f"Applies periodic noise suppression to array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            new_array = self._suppress_periodic_patterns(array)
            return new_array

    def postprocess(self, array: ArrayLike):

        if not self.do_postprocess:
            return array

        with lsection(
            f"Reapplies periodic noise to array of shape: {array.shape} and dtype: {array.dtype}"
        ):
            # turns out it is better to suppress the periodic patterns before and after:
            if self.post_processing_is_inverse:
                new_array = self._reapply_periodic_patterns(array)
            else:
                new_array = self._suppress_periodic_patterns(array)
            return new_array

    def _suppress_periodic_patterns(self, array: ArrayLike):
        self._original_dtype = array.dtype
        array = array.astype(numpy.float32, copy=False)

        # Compute Fourier intensity:
        ft = scipy.fft.fftn(array, workers=-1)
        ft = scipy.fft.fftshift(ft)
        spectrum = numpy.abs(ft).astype(numpy.float32)

        # Mask radius in pixels:
        mask_radius_pix = max(1, int(self.mask_radius * numpy.mean(array.shape)))

        # Denoise density:
        function, parameters, _ = calibrate_denoise_gaussian(
            spectrum, max_num_truncate=1
        )
        filtered_spectrum = function(spectrum, **parameters)

        # Minimum filtering:
        size = 2 * mask_radius_pix + 1
        filtered_spectrum = minimum_filter(filtered_spectrum, size=size)

        # Compute correction:
        correction = filtered_spectrum / spectrum

        # Coordinates of peaks to correct:
        self._coordinates = {}

        # Locate peaks:
        coordinates = peak_local_max(
            numpy.log1p(spectrum),
            min_distance=2 * mask_radius_pix,
            threshold_rel=self.threshold,
            num_peaks=self.max_num_peaks,
        )

        # center peak:
        center = tuple(s // 2 for s in array.shape)

        # For each peak:
        for i, coordinate in enumerate(coordinates):
            coordinate = tuple(coordinate)
            if coordinate != center:
                lprint(f"Suppressing peak at location: {coordinate}")

                # We compute a mask:
                mask = _sphere(
                    correction.shape, radius=mask_radius_pix, position=coordinate
                )

                # Apply the mask:
                factor = correction[mask]
                ft[mask] *= factor

                # keep coordinates:
                self._coordinates[i] = (mask, factor)

        # Transform back:
        ft = scipy.fft.ifftshift(ft)
        new_array = scipy.fft.ifftn(ft, workers=-1)
        new_array = numpy.real(new_array)

        # import napari
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_image(array, name='array')
        #     viewer.add_image(numpy.log1p(spectrum), name='spectrum')
        #     viewer.add_image(numpy.log1p(filtered_spectrum), name='filtered_spectrum')
        #     viewer.add_image(numpy.log1p(correction), name='correction')
        #     ft = scipy.fft.fftshift(ft)
        #     viewer.add_image(numpy.log1p(numpy.abs(ft).astype(numpy.float32)), name='spectrum_after')
        #     viewer.add_image(new_array, name='new_array')

        return new_array

    def _reapply_periodic_patterns(self, array: ArrayLike):
        array = array.astype(numpy.float32, copy=False)
        ft = scipy.fft.fftn(array, workers=-1)
        ft = scipy.fft.fftshift(ft)
        for mask, factor in self._coordinates.values():
            ft[mask] /= factor
        ft = scipy.fft.ifftshift(ft)
        new_array = scipy.fft.ifftn(ft, workers=-1)
        new_array = numpy.real(new_array)
        # cast back to original dtype:
        new_array = new_array.astype(self._original_dtype, copy=False)
        return new_array


def _sphere(shape, radius, position):
    # From : https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array/46626448

    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * len(shape)

    # generate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = numpy.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = numpy.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0
