Apply variance stabilisation transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the following lines to apply variance stabilisation transform on
a single image using our transforms API.

.. code-block:: python

   from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform

   vst = VarianceStabilisationTransform(mode="anscomb")

   preprocessed = vst.preprocess(image)
   postprocessed = vst.postprocess(preprocessed)

