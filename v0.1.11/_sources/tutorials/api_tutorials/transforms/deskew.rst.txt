Apply Deskew transform
~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply deskew transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.deskew import DeskewTransform

    deskew_transform = DeskewTransform(delta=-3, z_axis=0, skew_axis=1)

    deskewed_image = deskew_transform.preprocess(image)
    skewed_image = deskew_transform.postprocess(deskewed_image)

