Apply motion stabilisation transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply motion stabilisation transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.motion import MotionStabilisationTransform

   motion_transform = MotionStabilisationTransform(axes=0)

   preprocessed_image = motion_transform.preprocess(image.copy())
   postprocessed_image = motion_transform.postprocess(preprocessed_image.copy())

