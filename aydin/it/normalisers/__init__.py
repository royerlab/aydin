"""Value and shape normalization utilities for image translators.

Provides normaliser classes for mapping image values to a standard [0, 1]
range (MinMax, Percentile, Identity) and for normalizing image shapes by
handling batch and channel dimension permutations.
"""
