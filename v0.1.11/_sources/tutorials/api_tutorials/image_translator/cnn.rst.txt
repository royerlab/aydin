Denoising an image using ImageTranslatorCNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use the following lines to denoise a single image with default
options using our Object-Oriented denoising API.

.. code-block:: python

   from aydin.it.cnn import ImageTranslatorCNN

   it = ImageTranslatorCNN()

   it.train(noisy, noisy)
   denoised = it.translate(noisy)



It is also easy to pass specific transforms to use before and/or after
denoising. One can do the the following:

.. code-block:: python

   from aydin.it.cnn import ImageTranslatorCNN

   it = ImageTranslatorCNN()
   it.add_transform(RangeTransform())
   it.add_transform(PaddingTransform())

   it.train(noisy, noisy)
   denoised = it.translate(noisy)

