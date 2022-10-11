Apply periodic noise suppression transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply periodic noise suppression transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.periodic import PeriodicNoiseSuppressionTransform

   periodic_noise_suppression_transform = PeriodicNoiseSuppressionTransform()

   preprocessed = periodic_noise_suppression_transform.preprocess(image)
   postprocessed = periodic_noise_suppression_transform.postprocess(preprocessed)

