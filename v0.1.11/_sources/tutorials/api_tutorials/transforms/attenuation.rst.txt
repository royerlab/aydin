Apply Attenuation transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply attenuation transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.attenuation import AttenuationTransform

   attenuation_transform = AttenuationTransform(axes=0)

   preprocessed = attenuation_transform.preprocess(image)
   postprocessed = attenuation_transform.postprocess(preprocessed)

