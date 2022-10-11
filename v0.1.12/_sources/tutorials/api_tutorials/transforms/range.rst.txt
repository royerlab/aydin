Apply range transform
~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply range transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.range import RangeTransform

   range_transform = RangeTransform(mode="minmax")

   preprocessed = range_transform.preprocess(image)
   postprocessed = range_transform.postprocess(preprocessed)

