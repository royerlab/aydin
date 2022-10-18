Apply high pass transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply high pass transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.highpass import HighpassTransform

   highpass_transform = HighpassTransform()

   preprocessed = highpass_transform.preprocess(image)
   postprocessed = highpass_transform.postprocess(preprocessed)

