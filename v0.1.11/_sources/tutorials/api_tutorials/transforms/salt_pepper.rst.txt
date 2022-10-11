Apply salt-pepper transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply salt-pepper transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.salt_pepper import SaltPepperTransform

   salt_pepper_transform = SaltPepperTransform()

   corrected = salt_pepper_transform.preprocess(image)

