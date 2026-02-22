"""Default configuration values for classic denoiser calibration.

Provides named default values for optimiser settings, blind spots,
J-invariance interpolation mode, crop sizes, and maximum evaluation counts
used by the classic denoiser calibration functions.
"""

from collections import namedtuple

DefaultValue = namedtuple(typename="DefaultValue", field_names=["value"])
"""Named tuple wrapping a single default value.

Attributes
----------
value
    The default value.
"""

default_optimiser = DefaultValue('fast')
"""DefaultValue : Default optimiser strategy ('fast')."""

default_blind_spots = DefaultValue(None)
"""DefaultValue : Default blind-spot list (None, meaning standard single-pixel)."""

default_jinv_interpolation_mode = DefaultValue('gaussian')
"""DefaultValue : Default J-invariance interpolation mode ('gaussian')."""

default_crop_size_normal = DefaultValue(96000)
"""DefaultValue : Normal crop size in voxels (96 000)."""

default_crop_size_large = DefaultValue(128000)
"""DefaultValue : Large crop size in voxels (128 000)."""

default_crop_size_verylarge = DefaultValue(256000)
"""DefaultValue : Very large crop size in voxels (256 000)."""

default_crop_size_superlarge = DefaultValue(1000000)
"""DefaultValue : Super-large crop size in voxels (1 000 000)."""

default_max_evals_hyperlow = DefaultValue(16)
"""DefaultValue : Hyper-low maximum number of evaluations (16)."""

default_max_evals_ultralow = DefaultValue(32)
"""DefaultValue : Ultra-low maximum number of evaluations (32)."""

default_max_evals_verylow = DefaultValue(64)
"""DefaultValue : Very-low maximum number of evaluations (64)."""

default_max_evals_low = DefaultValue(128)
"""DefaultValue : Low maximum number of evaluations (128)."""

default_max_evals_normal = DefaultValue(256)
"""DefaultValue : Normal maximum number of evaluations (256)."""

default_max_evals_high = DefaultValue(512)
"""DefaultValue : High maximum number of evaluations (512)."""
