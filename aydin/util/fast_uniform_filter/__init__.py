"""Fast uniform (box) filter implementations.

This subpackage provides multiple implementations of the uniform (box)
filter: Numba CPU, Numba GPU, parallel SciPy, and an auto-selecting
dispatcher that picks the best method based on filter size.
"""
