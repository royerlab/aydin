from collections import namedtuple

DefaultValue = namedtuple(typename="DefaultValue", field_names=["value"])

default_optimiser = DefaultValue('fast')
default_blind_spots = DefaultValue(None)
default_jinv_interpolation_mode = DefaultValue('gaussian')
default_crop_size_normal = DefaultValue(96000)
default_crop_size_large = DefaultValue(128000)
default_crop_size_verylarge = DefaultValue(256000)
default_crop_size_superlarge = DefaultValue(1000000)
default_max_evals_hyperlow = DefaultValue(16)
default_max_evals_ultralow = DefaultValue(32)
default_max_evals_verylow = DefaultValue(64)
default_max_evals_low = DefaultValue(128)
default_max_evals_normal = DefaultValue(256)
default_max_evals_high = DefaultValue(512)
