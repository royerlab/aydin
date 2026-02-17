"""Salt-and-pepper noise correction transform.

Detects and corrects impulse noise (salt-and-pepper) using two complementary
methods: identifying over-represented pixel values and enforcing Lipschitz
continuity. This preprocessing step alleviates the denoising task for
subsequent algorithms.
"""

import numpy
from numpy import sort
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter

from aydin.it.classic_denoisers.lipschitz import denoise_lipschitz
from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import aprint, asection


class SaltPepperTransform(ImageTransformBase):
    """Salt-and-pepper noise correction transform.

    Detectors such as cameras have 'broken' pixels that blink, are very dim,
    or very bright. Other phenomena cause voxels to have very different
    values from their neighbors, this is often called
    <a href='https://en.wikipedia.org/wiki/Salt-and-pepper_noise'>impulse or 'salt-and-pepper' noise</a>.
    While self-supervised denoising can solve many of these issues, there is
    no reason not to alleviate the task, especially when there are simple and
    fast approaches that can tackle this kind of noise. This preprocessing
    uses two complementary methods: it identifies over-represented pixel
    values (stuck pixels) and enforces
    <a href='https://en.wikipedia.org/wiki/Lipschitz_continuity'>Lipschitz continuity</a>
    to detect outliers based on the local second-derivative of the image.
    Offending voxels are replaced with the median of their neighbors.
    Increase the threshold parameter to tolerate more variation, decrease it
    to be more aggressive. The algorithm is iterative, starting with the most
    offending pixels, until no pixels are corrected. You can set the max
    proportion of pixels that are allowed to be corrected if you can give a
    good estimate for that.
    <notgui>
    """

    preprocess_description = (
        "Salt and pepper pixels correction" + ImageTransformBase.preprocess_description
    )
    postprocess_description = "Not supported (why would anyone want to do that? ☺)"
    postprocess_supported = False
    postprocess_recommended = False

    def __init__(
        self,
        fix_repeated: bool = True,
        max_repeated: int = 4,
        fix_lipschitz: bool = True,
        lipschitz: float = 0.1,
        percentile: float = 0.01,
        num_iterations: int = 64,
        priority: float = 0.08,
        **kwargs,
    ):
        """Construct a SaltPepperTransform.

        Parameters
        ----------

        fix_repeated: bool
             Removes Salt & pepper by finding highly repeated values.
             These values are then considered as erroneous and are fixed
             by interpolation.

        max_repeated: int
            Max number of repeated values to fix.

        fix_lipschitz: bool
            Removes Salt & pepper by enforcing Lipschitz continuity.

        lipschitz : float
            Lipschitz threshold. Increase to tolerate more variation, decrease to be
            more aggressive in removing impulse/salt&pepper noise.

        percentile : float
            Percentile value used to determine the threshold
            for choosing the worst offending voxels per iteration
            according to the Lipschitz threshold.

        num_iterations : int
            Number of iterations for enforcing Lipschitz continuity.

        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocessing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)

        self.fix_lipschitz = fix_lipschitz
        self.num_iterations = num_iterations
        self.correction_percentile = percentile
        self.lipschitz = lipschitz
        self.fix_repeated = fix_repeated
        self.max_repeated = max_repeated

        self._original_dtype = None

        aprint(f"Instantiating: {self}")

    def __getstate__(self):
        """Return picklable state, excluding transient fields.

        Returns
        -------
        dict
            Object state without ``_original_dtype``.
        """
        state = self.__dict__.copy()
        del state['_original_dtype']
        return state

    def __str__(self):
        """Return a human-readable string representation.

        Returns
        -------
        str
            String showing the class name and key parameters.
        """
        return (
            f'{type(self).__name__} (fix_lipschitz={self.fix_lipschitz},'
            f' num_iterations={self.num_iterations},'
            f' correction_percentile={self.correction_percentile},'
            f' lipschitz={self.lipschitz},'
            f' fix_repeated={self.fix_repeated},'
            f' max_repeated={self.max_repeated} )'
        )

    def __repr__(self):
        """Return a detailed string representation.

        Returns
        -------
        str
            Same as ``__str__``.
        """
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        """Correct salt-and-pepper noise in the image.

        Applies two methods in sequence if enabled: repeated-value
        correction and Lipschitz continuity enforcement.

        Parameters
        ----------
        array : ArrayLike
            Input image array.

        Returns
        -------
        numpy.ndarray
            Corrected image as float32.
        """

        with asection(
            f"Broken Pixels Correction for array of shape: "
            f"{array.shape} and dtype: {array.dtype}:"
        ):
            # We save the original dtype:
            self._original_dtype = array.dtype

            # If needed, we convert to float32:
            array = array.astype(dtype=numpy.float32, copy=True)

            # First we look at over represented voxel
            # values -- a sign of problematic voxels,
            # and try to fix them:
            if self.fix_repeated:
                array = self._repeated_value_method(array)

            # Then we enforce Lipschitz continuity:
            if self.fix_lipschitz:
                array = self._lipschitz_method(array)

            return array

    def postprocess(self, array: ArrayLike):
        """Convert back to original data type (no inverse correction).

        Salt-and-pepper correction is not reversible, so this method
        only performs a dtype cast.

        Parameters
        ----------
        array : ArrayLike
            Denoised image array.

        Returns
        -------
        numpy.ndarray
            Image cast to the original data type.
        """
        # undoing this transform is impractical and unlikely to be useful
        array = array.astype(self._original_dtype, copy=False)
        return array

    def _repeated_value_method(self, array: ArrayLike):
        """Correct pixels whose values are over-represented in the image.

        Identifies pixel values that appear far more frequently than
        expected under a uniform distribution, treats them as erroneous,
        and replaces them by iterative uniform-filter interpolation from
        surrounding good pixels.

        Parameters
        ----------
        array : ArrayLike
            Input image array (float32).

        Returns
        -------
        numpy.ndarray
            Corrected array with over-represented values replaced.
        """
        with asection(
            "Correcting for wrong pixels values using the 'repeated-value' approach:"
        ):
            unique, counts = numpy.unique(array, return_counts=True)

            # How many unique values in image?
            num_unique_values = unique.size
            aprint(f"Number of unique values in image: {num_unique_values}.")

            # Most occuring value
            most_occuring_value = unique[numpy.argmax(counts)]
            highest_count = numpy.max(counts)
            aprint(
                f"Most occurring value in array: "
                f"{most_occuring_value}, "
                f"{highest_count} times."
            )

            # Assuming a uniform distribution we would expect
            # each value to be used at most:
            average_count = array.size // num_unique_values
            aprint(
                f"Average number of occurences of a value "
                f"assuming uniform distribution: "
                f"{average_count}"
            )

            # We fix at most n over-represented values:
            selected_counts = sort(counts.flatten())

            # First we ignore counts below a certain thresholds:
            selected_counts = selected_counts[selected_counts > average_count]

            # use Otsu split to clean up remaining values:
            mask = _otsu_split(selected_counts)
            selected_counts = selected_counts[mask]

            # Maximum number of repeated values to remove:
            n = self.max_repeated
            n = min(n, len(selected_counts))
            max_tolerated_count = selected_counts[-n]
            aprint(f"Maximum tolerated count per value: {max_tolerated_count}.")

            # If a voxel value appears over more than 0.1%
            # of voxels, then it is a problematic value:
            problematic_counts_mask = counts > max_tolerated_count
            problematic_counts = counts[problematic_counts_mask]
            problematic_values = unique[problematic_counts_mask]

            aprint(f"Problematic values: {list(problematic_values)}.")
            aprint(f"Problematic counts: {list(problematic_counts)}.")

            # We construct the mask of good values:
            good_values_mask = numpy.ones_like(array, dtype=numpy.bool_)
            for problematic_value in problematic_values:
                good_values_mask &= array != problematic_value

            with asection(f"Correcting voxels with values: {problematic_values}."):

                # We save the good values (copy!):
                good_values = array[good_values_mask].copy()

                # We compute the number of iterations:
                num_bad_values = array.size - len(good_values)
                num_iterations = 16 * int(
                    (array.size / num_bad_values) ** (1.0 / array.ndim)
                )

                # We solve the harmonic equation:
                for i in range(num_iterations):
                    aprint(f"Iteration {i}")
                    # We compute the median:
                    array = uniform_filter(array, size=3)
                    # We use the median to correct pixels:
                    array[good_values_mask] = good_values

                # count number of corrections for this round:
                num_corrections = numpy.sum(mask)
                aprint(f"Number of corrections: {num_corrections}.")

        return array

    def _lipschitz_method(self, array):
        """Correct salt-and-pepper noise by enforcing Lipschitz continuity.

        Iteratively replaces voxels that violate the Lipschitz continuity
        constraint with smoothed values from their neighborhood.

        Parameters
        ----------
        array : numpy.ndarray
            Input image array (float32).

        Returns
        -------
        numpy.ndarray
            Corrected array with Lipschitz continuity enforced.
        """
        # Iterations:
        with asection(
            "Correcting for wrong pixels values using the Lipschitz approach:"
        ):
            array = denoise_lipschitz(
                array,
                lipschitz=self.lipschitz,
                percentile=self.correction_percentile,
                max_num_iterations=self.num_iterations,
            )

            return array

            # OLD METHOD KEEP!
            # for i in range(self.num_iterations):
            #     aprint(f"Iteration {i}")
            #
            #     # Compute median:
            #     median = median_filter(array, size=3)
            #
            #     # We scale the lipschitz threshold to the image std at '3 sigma' :
            #     lipschitz = self.lipschitz * 3 * median.std()
            #
            #     # We compute the 'error':
            #     median, error = self._compute_error(
            #         array, median=median, lipschitz=lipschitz
            #     )
            #
            #     # We compute the threshold on the basis of the errors,
            #     # we first tackle the most offending voxels:
            #     threshold = numpy.percentile(
            #         error, q=100 * (1 - self.correction_percentile)
            #     )
            #
            #     # We compute the mask:
            #     mask = error > threshold
            #
            #     # count number of corrections for this round:
            #     num_corrections = numpy.sum(mask)
            #     aprint(f"Number of corrections: {num_corrections}")
            #
            #     # if no corrections made we stop iterating:
            #     if num_corrections == 0:
            #         break
            #
            #     # We keep track of the proportion of voxels corrected:
            #     proportion = (
            #         num_corrections + total_number_of_corrections
            #     ) / array.size
            #     aprint(
            #         f"Proportion of corrected pixels: "
            #         f"{int(proportion * 100)}% (up to now), "
            #         f"versus maximum: "
            #         f"{int(self.max_proportion_corrected * 100)}%) "
            #     )
            #
            #     # If too many voxels have been corrected we stop:
            #     if proportion > self.max_proportion_corrected:
            #         break
            #
            #     # We use the median to correct pixels:
            #     array[mask] = median[mask]
            #
            #     # increment total number of corrections:
            #     total_number_of_corrections += num_corrections

    def _compute_error(self, array, median, lipschitz):
        """Compute the Lipschitz error map between an array and its median.

        Parameters
        ----------
        array : numpy.ndarray
            Original image array.
        median : numpy.ndarray
            Median-filtered version of the array.
        lipschitz : float
            Lipschitz threshold value.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            A tuple of (median, error) where error contains the
            thresholded absolute difference.
        """
        # we compute the error map:
        error = median.copy()
        error -= array
        numpy.abs(error, out=error)
        numpy.maximum(error, lipschitz, out=error)
        error -= lipschitz
        return median, error


def _otsu_split(array: ArrayLike):
    """Split an array into two classes using Otsu's method.

    Parameters
    ----------
    array : ArrayLike
        Input array of values to split.

    Returns
    -------
    numpy.ndarray
        Boolean mask where True indicates the upper class.
    """
    # Flatten array:
    shape = array.shape
    array = array.reshape(-1)

    mean_weigth = 1.0 / array.size
    his, bins = numpy.histogram(array, bins='auto', density=True)
    final_thresh = -1
    final_value = -1
    for i in range(1, len(bins) - 1):
        Wb = numpy.sum(his[:i]) * mean_weigth
        Wf = numpy.sum(his[i:]) * mean_weigth

        mub = numpy.mean(his[:i])
        muf = numpy.mean(his[i:])

        value = Wb * Wf * (mub - muf) ** 2

        # print("Wb", Wb, "Wf", Wf)
        # print("t", i, "value", value)

        if value > final_value:
            final_thresh = 0.5 * (bins[i] + bins[i + 1])
            final_value = value

    mask = array > final_thresh

    mask = mask.reshape(shape)

    return mask
