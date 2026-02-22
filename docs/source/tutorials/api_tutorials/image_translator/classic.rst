Denoising an image using ImageDenoiserClassic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use the following lines to denoise a single image with default
options using our Object-Oriented denoising API.

.. code-block:: python

   from aydin.it.classic import ImageDenoiserClassic

   it = ImageDenoiserClassic(method='butterworth')

   it.train(noisy, noisy)
   denoised = it.translate(noisy)



It is also easy to pass specific transforms to use before and/or after
denoising. One can do the following:

.. code-block:: python

   from aydin.it.classic import ImageDenoiserClassic
   from aydin.it.transforms.range import RangeTransform
   from aydin.it.transforms.padding import PaddingTransform

   it = ImageDenoiserClassic(method='butterworth')
   it.add_transform(RangeTransform())
   it.add_transform(PaddingTransform())

   it.train(noisy, noisy)
   denoised = it.translate(noisy)
