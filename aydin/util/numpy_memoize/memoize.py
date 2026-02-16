"""Memoization decorator that supports NumPy arrays.

Provides a single-entry cache decorator that works with unhashable
types like NumPy arrays by using object identity (``id``) for cache
key comparison.
"""

import functools

import numpy


class memoize_last(object):
    """Single-entry memoization decorator supporting NumPy arrays.

    Caches the most recent return value of a function or method based on
    its argument signatures. If any single argument changes, the function
    is re-evaluated. Supports NumPy arrays (which are not hashable) by
    comparing object identity via ``id()``.

    .. warning::
       Because NumPy array caching relies on ``id()``, the cache may
       produce stale results if an array is modified in-place or if a
       new array is allocated at the same memory address as a previous one.

    Attributes
    ----------
    value : object or None
        The cached return value.

    References
    ----------
    Original implementation from:
    https://gist.github.com/dpo/1222577
    https://github.com/numpy/numpy/issues/14294
    """

    def __init__(self, callable):
        """Initialize the memoizer.

        Parameters
        ----------
        callable : callable
            The function or method to memoize.
        """
        self._callable = callable
        self._callable_is_method = False
        self.value = None  # Cached value or derivative.
        self._args_hashes = {}
        return

    def __get_hash(self, x):
        """Compute a hash-like signature for an argument.

        For NumPy arrays, returns ``id(x)`` since arrays are not hashable.
        For all other types, returns ``hash(x)``.

        Parameters
        ----------
        x : object
            Argument to compute signature for.

        Returns
        -------
        int
            Hash or identity value.
        """
        if isinstance(x, numpy.ndarray):
            # _x = x.view(numpy.uint8)
            # return hash(hashlib.sha1(_x).hexdigest())
            return id(x)

        return hash(x)

    def __call__(self, *args, **kwargs):
        """Call the wrapped function, using cached result if arguments are unchanged.

        Parameters
        ----------
        *args : object
            Positional arguments.
        **kwargs : object
            Keyword arguments.

        Returns
        -------
        object
            Cached or freshly computed result.
        """
        # The callable will be called if any single argument is new or changed.

        callable = self._callable
        evaluate = False

        # If we're memoizing a class method, the first argument will be 'self'
        # and need not be memoized.
        firstarg = 1 if self._callable_is_method else 0

        # Get signature of all arguments.
        nargs = callable.__code__.co_argcount  # Non-keyword arguments.
        argnames = callable.__code__.co_varnames[firstarg:nargs]
        argvals = args[firstarg:]

        allargs = dict(zip(argnames, argvals)) | kwargs

        for argname, argval in allargs.items():

            _arg_hash = self.__get_hash(argval)

            try:
                cached_arg_hash = self._args_hashes[argname]
                if cached_arg_hash != _arg_hash:
                    self._args_hashes[argname] = _arg_hash
                    evaluate = True

            except KeyError:
                self._args_hashes[argname] = _arg_hash
                evaluate = True

        # If cache invalid, recompute value:
        if evaluate:
            self.value = callable(*args, **kwargs)

        return self.value

    def __get__(self, obj, *args):
        """Support instance methods."""
        self._callable_is_method = True
        return functools.partial(self.__call__, obj)

    def __repr__(self):
        """Return the wrapped function or method's docstring."""
        return self._callable.__doc__
