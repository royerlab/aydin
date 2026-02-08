"""Maskout layer for masking architectures in Keras/TensorFlow (deprecated)."""

import keras
from deprecated import deprecated
from keras.layers import Layer, Multiply


@deprecated(
    "All the Tensorflow related code and dependencies are deprecated and will be removed by v0.1.16"
)
class Maskout(Layer):
    """Keras layer that multiplies a mask with an image.

    Used in masking-based self-supervised training architectures to
    apply pixel masks to the network output.

    Parameters
    ----------
    kwargs
    """

    def __init__(self, **kwargs):
        super(Maskout, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the layer.

        Parameters
        ----------
        input_shape : list of tuple
            List of input shapes (image and mask shapes).
        """
        assert isinstance(input_shape, list)
        super(Maskout, self).build(input_shape)

    def call(self, x, **kwargs):
        """Apply element-wise multiplication of the inputs.

        Parameters
        ----------
        x : list of tf.Tensor
            List of two tensors: [image, mask].
        **kwargs
            Additional keyword arguments (e.g., ``training``).

        Returns
        -------
        tf.Tensor
            Element-wise product of the image and mask tensors.
        """
        assert isinstance(x, list)
        return Multiply()(x)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : list of tuple
            List of input shapes [shape_a, shape_b].

        Returns
        -------
        tuple
            Output shape, equal to the first input shape.
        """
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a
