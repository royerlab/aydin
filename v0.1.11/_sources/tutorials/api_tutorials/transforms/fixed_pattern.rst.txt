Apply fixed pattern transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply fixed pattern transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.fixedpattern import FixedPatternTransform

   fixedpattern_transform = FixedPatternTransform(axes=[1, 2])

   preprocessed = fixedpattern_transform.preprocess(image)

