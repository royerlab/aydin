How to implement Noise2Noise using ImageTranslatorFGR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is quite easy to train `Noise2Noise <https://github.com/NVlabs/noise2noise>`_  model with image translators provided
in Aydin API. You can see an example below using ImageTranslatorFGR:

.. code-block:: python

   from aydin.it.fgr import ImageTranslatorFGR

   it = ImageTranslatorFGR()
   it.add_transform(RangeTransform())
   it.add_transform(PaddingTransform())

   for noisy_image1, noisy_image2 in noisy_pairs:
       it.train(noisy_image1, noisy_image2)

   denoised = it.translate(noisy_image)


Noise2Noise is a great method to train a model from pairs of images sharing the same information signal with
different noise instances. You can find more information about how to prepare noisy image pairs on the original
paper. More information about the paper is available `here <https://arxiv.org/pdf/1803.04189.pdf>`_
and the code that is published with paper can be found on `github <https://github.com/NVlabs/noise2noise>`_.

