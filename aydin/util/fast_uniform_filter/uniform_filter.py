"""Auto-selecting uniform filter dispatch.

Automatically selects between Numba and parallel SciPy implementations
based on filter size for optimal performance.
"""

from aydin.util.fast_uniform_filter.numba_cpu_uf import numba_cpu_uniform_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.util.log.log import lprint


def uniform_filter_auto(image, size: int, printout_choice: bool = False):
    """Apply a uniform (box) filter using the fastest available method.

    For large filter sizes (> 128), uses the Numba CPU implementation
    which scales well. For smaller sizes, uses parallel SciPy which
    has lower overhead.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    size : int or tuple of int
        Filter size. If int, applied uniformly across all axes.
    printout_choice : bool
        If True, prints which implementation was selected.

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """

    # Ensures size parameter is integer:
    size = int(round(size))

    # Different methods perform differently based on filter size:
    max_size = max(size) if isinstance(size, tuple) else size
    if max_size > 128:
        # Numba scales well for large filter sizes:
        output = numba_cpu_uniform_filter(image, size=size, mode="nearest")
        if printout_choice:
            lprint(f"Computed filter of size: {size} with Numba")
    else:
        # Scipy parallel is more efficient for small filter sizes:
        output = parallel_uniform_filter(image, size=size, mode="nearest")
        if printout_choice:
            lprint(f"Computed filter of size: {size} with parallel scipy")

    return output
