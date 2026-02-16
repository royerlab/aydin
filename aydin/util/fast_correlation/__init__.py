"""Fast n-dimensional correlation implementations.

This subpackage provides accelerated correlation functions using Numba
CPU parallelism, with fallback to SciPy. Also includes a parallel
tiled correlation approach using joblib for multi-threaded execution.
"""
