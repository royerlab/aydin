"""Utilities for mapping napari layer axis metadata to Aydin FileMetadata."""

import numpy

from aydin.io.io import FileMetadata, is_batch, is_channel

# Napari axis label -> Aydin axis code mapping
_NAPARI_LABEL_TO_AYDIN = {
    't': 'T',
    'time': 'T',
    'z': 'Z',
    'depth': 'Z',
    'y': 'Y',
    'row': 'Y',
    'x': 'X',
    'col': 'X',
    'column': 'X',
    'c': 'C',
    'channel': 'C',
    'ch': 'C',
}


def _labels_are_generic(labels):
    """Check whether napari axis labels are the default unnamed placeholders.

    Default napari labels are integer-indexed strings like '0', '1', '2', ...
    or empty strings.

    Parameters
    ----------
    labels : tuple of str
        Axis labels from ``viewer.dims.axis_labels``.

    Returns
    -------
    bool
        True if all labels appear to be unnamed / generic.
    """
    for label in labels:
        s = str(label).strip().lower()
        if s and s not in {str(i) for i in range(len(labels))}:
            return False
    return True


def _guess_axes_from_shape(shape):
    """Heuristic axis assignment based on array shape alone.

    Rules:
    - 2D -> YX
    - 3D -> ZYX (unless last dim <= 8, then YXC)
    - 4D -> TZYX (unless last dim <= 8, then ZYXC)
    - 5D -> TZYXC (unless last dim > 8, then QTZYX)
    - 6D+ -> Q...TZYX

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.

    Returns
    -------
    str
        A string of axis codes with one code per dimension.
    """
    ndim = len(shape)
    if ndim <= 1:
        return 'X' * ndim
    if ndim == 2:
        return 'YX'
    if ndim == 3:
        if shape[-1] <= 8:
            return 'YXC'
        return 'ZYX'
    if ndim == 4:
        if shape[-1] <= 8:
            return 'ZYXC'
        return 'TZYX'
    if ndim == 5:
        if shape[-1] <= 8:
            return 'TZYXC'
        return 'QTZYX'
    # 6D+: pad with Q's
    extra = ndim - 4
    return 'Q' * extra + 'TZYX'


def detect_axes_from_napari_layer(layer, viewer=None):
    """Detect Aydin axis metadata from a napari image layer.

    Attempts to map napari ``viewer.dims.axis_labels`` to Aydin axis codes.
    Falls back to shape-based heuristics when labels are generic (unnamed).

    Parameters
    ----------
    layer : napari.layers.Image
        The napari image layer.
    viewer : napari.Viewer, optional
        The napari viewer (used to read ``dims.axis_labels``).
        If None, shape heuristics are used.

    Returns
    -------
    FileMetadata
        An Aydin ``FileMetadata`` instance populated with axes, shape,
        dtype, batch_axes, and channel_axes.
    """
    shape = layer.data.shape

    axes = None

    # Try to get labels from viewer
    if viewer is not None:
        try:
            labels = viewer.dims.axis_labels
            if labels and len(labels) == len(shape) and not _labels_are_generic(labels):
                codes = []
                for label in labels:
                    code = _NAPARI_LABEL_TO_AYDIN.get(str(label).strip().lower())
                    if code is None:
                        code = 'Q'
                    codes.append(code)
                axes = ''.join(codes)
        except Exception:
            pass

    # Fallback to shape heuristics
    if axes is None:
        axes = _guess_axes_from_shape(shape)

    # Build FileMetadata
    metadata = FileMetadata()
    metadata.axes = axes
    metadata.shape = shape
    metadata.dtype = numpy.dtype(layer.data.dtype)
    metadata.format = 'napari'

    metadata.batch_axes = tuple(is_batch(code, shape, axes) for code in axes)
    metadata.channel_axes = tuple(
        is_channel(code, shape[i]) for i, code in enumerate(axes)
    )

    return metadata
