import functools
import numpy


class memoize_last(object):
    """Decorator class used to cache the most recent value of a function
    or method based on the signature of its arguments. If any single
    argument changes, the function or method is evaluated afresh.

    Importantly: works even for numpy arrays although they are not hashable.
    We use the object id, so great care has to be taken when using this!!

    Original implementation from:
    https://gist.github.com/dpo/1222577
    https://github.com/numpy/numpy/issues/14294

    """

    def __init__(self, callable):
        self._callable = callable
        self._callable_is_method = False
        self.value = None  # Cached value or derivative.
        self._args_hashes = {}
        return

    def __get_hash(self, x):
        # Return signature of argument.
        if isinstance(x, numpy.ndarray):
            # _x = x.view(numpy.uint8)
            # return hash(hashlib.sha1(_x).hexdigest())
            return id(x)

        return hash(x)

    def __call__(self, *args, **kwargs):
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

        for (argname, argval) in allargs.items():

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

        print(f"{tuple(f'{k}:{id(v)}' for k,v in allargs.items())} -> {id(self.value)}")

        return self.value

    def __get__(self, obj, objtype):
        "Support instance methods."
        self._callable_is_method = True
        return functools.partial(self.__call__, obj)

    def __repr__(self):
        "Return the wrapped function or method's docstring."
        return self.method.__doc__
