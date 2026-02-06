"""Data histogram balancer for training data sampling.

This module provides `DataHistogramBalancer`, a utility that prevents
over-representation of certain pixel intensities in the training data
by balancing the histogram of sampled data entries.
"""

import math
import random

import numpy
from skimage.transform import downscale_local_mean

from aydin.util.log.log import aprint, asection


class DataHistogramBalancer:
    """Training data histogram balancer.

    Prevents over-representation of certain pixel intensities in the training
    data by maintaining a histogram of accepted entries and limiting the number
    of entries per histogram bin. Supports optional favouring of bright or dark
    pixels via a linear weighting slope.
    """

    def __init__(
        self,
        number_of_bins: int = 1024,
        keep_ratio: float = 1,
        balance: bool = True,
        use_median: bool = False,
        favour_bright_pixels: float = 0,
    ):
        """Construct a DataHistogramBalancer.

        Parameters
        ----------
        number_of_bins : int
            Number of histogram bins for intensity tracking.
        keep_ratio : float
            Target ratio of data entries to keep (0 to 1).
        balance : bool
            If True, enable histogram-based balancing. If False,
            use uniform random decimation.
        use_median : bool
            If True, use median for batch intensity estimation.
            If False, use mean.
        favour_bright_pixels : float
            Value within [-1.0, 1.0] controlling brightness bias.
            Positive values allow more bright pixels; negative values
            allow more dark pixels. Zero treats all intensities equally.
        """

        self.favour_bright_pixels = favour_bright_pixels
        self.use_median = use_median
        self.balance = balance

        self.number_of_bins = number_of_bins
        self.keep_ratio = keep_ratio

        self.min_value = 0
        self.max_value = 1

    def calibrate(
        self, array, batch_length, percentile=0.000001, num_batches_to_sample=128 * 1024
    ):
        """Calibrate the histogram balancer by estimating the data value range.

        Samples random batches from the array to estimate the effective
        min and max values for histogram binning.

        Parameters
        ----------
        array : numpy.ndarray
            Flat array of all target values to calibrate on.
        batch_length : int
            Length of each random batch sample.
        percentile : float
            Percentile for clipping extreme values. Default is 0.000001.
        num_batches_to_sample : int
            Number of random batches to sample for range estimation.
        """

        array = array.ravel()

        # Calibration is not needed if we don't balance
        if not self.balance:
            aprint("Balancer: no calibration needed ")
            self.total_kept_counter = 0
            return

        # Find lower and upper bounds:
        if percentile > 0:
            full_min_value = numpy.percentile(array, 100 * percentile)
            full_max_value = numpy.percentile(array, 100 - 100 * percentile)
        else:
            full_min_value = array.min()
            full_max_value = array.max()

        with asection("Balancer: sampling to estimate min and max values for entries"):
            # refine with sampling
            values_list = []
            array_length = array.size
            for i in range(num_batches_to_sample):
                start = random.randint(0, array_length - 1 - batch_length)
                stop = start + batch_length

                batch = array[..., start:stop]
                if self.use_median:
                    intensity = numpy.median(batch)
                else:
                    intensity = numpy.mean(batch)

                values_list.append(intensity)

            batch_min_value = min(values_list)
            batch_max_value = max(values_list)

        # Range:
        minmax_range = batch_max_value - batch_min_value

        # We extend the range by 5% on both sides to account for possible errors of the range estimate:
        self.min_value = batch_min_value - 0.05 * minmax_range
        self.max_value = batch_max_value + 0.05 * minmax_range

        # Keep the batch min and max within the full min and max (can happen because of percentile...)
        batch_min_value = max(full_min_value, batch_min_value)
        batch_max_value = min(full_max_value, batch_max_value)

        aprint(
            f"Balancer: full data min and max: [{full_min_value}, {full_max_value}] "
        )
        aprint(f"Balancer: batch min and max: [{batch_min_value}, {batch_max_value}] ")
        aprint(
            f"Balancer: effective min and max: [{self.min_value}, {self.max_value}] "
        )

    def initialise(self, total_entries):
        """Initialize the histogram counters for a new sampling session.

        Must be called before `add_entry`. Sets up histogram bins and
        computes per-bin entry limits based on the keep ratio.

        Parameters
        ----------
        total_entries : int
            Total number of entries that will be offered to the balancer.
        """
        self.histogram_kept = numpy.zeros(self.number_of_bins)
        self.histogram_all = numpy.zeros(self.number_of_bins)

        self.total_entries = total_entries

        max_density = int(
            math.ceil(self.keep_ratio * self.total_entries / self.number_of_bins)
        )

        if self.favour_bright_pixels > 0:
            beta = self.favour_bright_pixels
            n = self.number_of_bins
            self.max_entries_per_bin = [
                2 * beta * max_density * (i / n) + (1 - beta) * max_density
                for i in range(self.number_of_bins)
            ]
        else:
            self.max_entries_per_bin = [max_density] * self.number_of_bins

        # In the case of no balancing, we just keep track of the kept entries:
        self.total_kept_counter = 0

    def add_entry(self, array):
        """Offer a data entry to the balancer and determine if it should be kept.

        The balancer decides whether to keep the entry based on the
        histogram bin occupancy and keep ratio.

        Parameters
        ----------
        array : numpy.ndarray
            Data entry array (will be raveled for intensity computation).

        Returns
        -------
        bool
            True if the entry should be kept, False if it should be dropped.
        """

        array = array.ravel()

        # This ensures that we stay within the bounds of the 'keep-ratio'
        fill_ratio = self.total_kept_counter / self.total_entries
        if fill_ratio > self.keep_ratio:
            return False

        if self.balance:

            if self.use_median:
                intensity = numpy.median(array)
            else:
                intensity = numpy.mean(array)

            # Rescale to range [0,1] range:
            intensity = (intensity - self.min_value) / (self.max_value - self.min_value)
            intensity = min(1, max(0, intensity))

            index = int(self.number_of_bins * intensity)
            index = min(len(self.histogram_kept) - 1, index)

            self.histogram_all[index] += 1

            if self.histogram_kept[index] < self.max_entries_per_bin[index]:
                self.histogram_kept[index] += 1
                self.total_kept_counter += 1
                return True

            return False

        else:
            # we just decimate uniformly according to the keep-ratio without histogram balancing:
            if random.random() < self.keep_ratio:
                self.total_kept_counter += 1
                return True

            return False

    def get_histogram_kept_as_string(self):
        """Returns the kept part of histogram as a string

        Returns
        -------
        str

        """

        if not self.balance:
            return '│ -- no balancing -- │'
        else:
            histogram_all = downscale_local_mean(
                self.histogram_all, factors=(8,), cval=0
            )
            maxvalue = numpy.max(histogram_all)
            histogram = downscale_local_mean(self.histogram_kept, factors=(8,), cval=0)
            return (
                '│'
                + ''.join((_value_to_fill_char(x / maxvalue)) for x in histogram)
                + '│'
            )

    def get_histogram_all_as_string(self):
        """Returns the whole histogram as a string

        Returns
        -------
        str

        """

        if not self.balance:
            return '│ -- no balancing -- │'
        else:
            histogram = downscale_local_mean(self.histogram_all, factors=(8,), cval=0)
            maxvalue = numpy.max(histogram)
            return (
                '│'
                + ''.join((_value_to_fill_char(x / maxvalue)) for x in histogram)
                + '│'
            )

    def get_histogram_dropped_as_string(self):
        """Returns the dropped part of histogram as a string

        Returns
        -------
        str

        """

        if not self.balance:
            return '│ -- no balancing -- │'
        else:
            histogram_all = downscale_local_mean(
                self.histogram_all, factors=(8,), cval=0
            )
            maxvalue = numpy.max(histogram_all)
            histogram_dropped = self.histogram_all - self.histogram_kept
            histogram = downscale_local_mean(histogram_dropped, factors=(8,), cval=0)
            return (
                '│'
                + ''.join((_value_to_fill_char(x / maxvalue)) for x in histogram)
                + '│'
            )

    def total_kept(self):
        """Return the total number of entries kept so far.

        Returns
        -------
        int
            Number of kept entries.
        """
        return (
            int(self.histogram_kept.sum()) if self.balance else self.total_kept_counter
        )

    def percentage_kept(self):
        """Return the fraction of entries kept relative to total offered.

        Returns
        -------
        float
            Fraction of entries kept (0 to 1).
        """
        if self.balance:
            return min(1, self.histogram_kept.sum() / self.total_entries)
        else:
            return self.total_kept_counter / self.total_entries


def _value_to_fill_char(x):
    """Convert a normalized value to a Unicode fill character for histogram display.

    Parameters
    ----------
    x : float
        Normalized value between 0 and 1.

    Returns
    -------
    str
        A single Unicode character representing the fill level.
    """
    if x <= 0.00:
        return ' '
    elif x <= 0.01:
        return '·'
    elif x <= 0.05:
        return '-'
    elif x <= 0.10:
        return '■'
    elif x <= 0.25:
        return '░'
    elif x <= 0.50:
        return '▒'
    elif x <= 0.75:
        return '▓'
    elif x > 0.75:
        return '█'
