Apply padding transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply padding transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.padding import PaddingTransform

   padding_transform = PaddingTransform(pad_width=17)

   preprocessed = padding_transform.preprocess(image)
   postprocessed = padding_transform.postprocess(preprocessed)

