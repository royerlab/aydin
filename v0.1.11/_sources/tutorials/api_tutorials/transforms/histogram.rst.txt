Apply histogram equalisation transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply histogram equalisation
transform on a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.histogram import HistogramEqualisationTransform

   histogram_transform = HistogramEqualisationTransform()

   preprocessed = histogram_transform.preprocess(image)
   postprocessed = histogram_transform.postprocess(preprocessed)

