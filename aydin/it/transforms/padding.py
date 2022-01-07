import numpy

from numpy.typing import ArrayLike

from aydin.it.transforms.base import ImageTransformBase
from aydin.util.log.log import lsection, lprint


class PaddingTransform(ImageTransformBase):
    """Padding (and then Cropping)

    Adds a border to the image before denoising and then removes that border
    from the denoised image. This reduces border artifacts when denoising
    certain images. In the case of self-supervised blind-spot based denoisers
    (e.g. N2S) padding must be carefully chosen because the added pixels
    should not 'give away' the value of their neighbors. Thus, not all
    padding modes are recommended. 'symmetric' is the recommended default.

    Other supported modes: 'constant': Pads with a constant value of 0,
    'linear_ramp': Pads with the linear ramp between end_value and the array
    edge value, 'maximum': Pads with the maximum value of all or part of the
    vector along each axis, 'mean': Pads with the mean value of all or part
    of the vector along each axis, 'median': Pads with the median value of
    all or part of the vector along each axis, 'minimum': Pads with the
    minimum value of all or part of the vector along each axis.
    """

    preprocess_description = "Pads image" + ImageTransformBase.preprocess_description
    postprocess_description = "Crops image" + ImageTransformBase.postprocess_description
    postprocess_supported = True
    postprocess_recommended = True

    def __init__(
        self,
        pad_width: int = 3,
        mode: str = 'reflect',
        min_length_to_pad: int = 8,
        priority: float = 0.9,
        **kwargs,
    ):
        """
        Constructs a Padding Transform

        Parameters
        ----------
        pad_width : int
            Amount of padding on all sides of the array
        mode : str
            Padding mode, may be: 'constant', 'linear_ramp', 'maximum',
            'mean', 'median', 'minimum', 'symmetric', 'reflect'
        min_length_to_pad : int
            Minimal dimension length to pad. This avoids padding for 'channel-like'
            dimensions for which it does not make sense to pad.
        priority : float
            The priority is a value within [0,1] used to determine the order in
            which to apply the pre- and post-processing transforms. Transforms
            are sorted and applied in ascending order during preprocesing and in
            the reverse, descending, order during post-processing.
        """
        super().__init__(priority=priority, **kwargs)
        self.pad_width = pad_width
        self.mode = mode
        self.min_length_to_pad = min_length_to_pad
        self._pad_width = None

        lprint(f"Instanciating: {self}")

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_pad_width']
        return state

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f' (pad_width={self.pad_width},'
            f' min_length_to_pad={self.min_length_to_pad},'
            f' mode={self.mode})'
        )

    def __repr__(self):
        return self.__str__()

    def preprocess(self, array: ArrayLike):
        with lsection(
            f"Padding array of shape: {array.shape} with {self.pad_width} voxels and mode: {self.mode}:"
        ):
            padded_array, self._pad_width = _pad(
                array, self.mode, self.pad_width, self.min_length_to_pad
            )
            return padded_array

    def postprocess(self, array: ArrayLike):
        if not self.do_postprocess:
            return array

        with lsection(
            f"Cropping array of shape: {array.shape} by removing padding of {self.pad_width} voxels:"
        ):
            new_array = _unpad(array, pad_width=self._pad_width)
            return new_array


def _pad(array: ArrayLike, mode: str, pad_width: int, min_length_to_pad: int):

    # Compute pad width:
    if isinstance(pad_width, int):
        pad_width = tuple(
            (pad_width, pad_width) if s >= min_length_to_pad else (0, 0)
            for s in array.shape
        )
    else:
        raise ValueError(
            f"Unsupported padding value, must be positive integer, was: {pad_width}"
        )

    # Do padding:
    lprint(f"Effective padding widths: {pad_width}")
    padded_array = numpy.pad(array, pad_width=pad_width, mode=mode)

    return padded_array, pad_width


def _unpad(array: ArrayLike, pad_width):

    # Compute slice:
    slices = []
    for before, after in pad_width:
        after = None if after == 0 else -after
        slices.append(slice(before, after))

    # Crop to 'unpad':
    lprint(f"Effective cropping widths: {pad_width}")
    cropped_array = array[tuple(slices)]

    return cropped_array
