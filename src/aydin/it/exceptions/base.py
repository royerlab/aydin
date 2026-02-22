"""Base exception classes for the Image Translator framework."""


class ArrayShapeDoesNotMatchError(Exception):
    """Raised when array shapes are incompatible.

    This error is raised when input and target image shapes do not match,
    or when axis specifications are inconsistent with image dimensions.
    """

    pass
