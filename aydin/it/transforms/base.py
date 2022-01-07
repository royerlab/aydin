from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class ImageTransformBase(ABC):
    """Image Transforms base class

    Transforms take an array, apply a transformation that is meant to
    facilitate denoising (or another image restoration task), and then
    optionally, offer a way to go back by undoing the transformation. For
    example, motion correction can be applied to facilitate denoising,
    and then after denoising has ben performed, the motion can be reapplied,
    if really needed.
    """

    preprocess_description = " (before denoising)"
    postprocess_description = " (after denoising)"
    postprocess_supported = None

    def __init__(self, priority: float = -1, do_postprocess: bool = True, **kwargs):
        """

        Parameters
        ----------
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.

        do_postprocess : bool When True, post-processing will occur,
        when False, no post-processing is done.

        kwargs
        """
        self.priority = priority
        self.do_postprocess = do_postprocess

    @abstractmethod
    def preprocess(self, array: ArrayLike):
        """ """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, array: ArrayLike):
        """ """
        raise NotImplementedError()
