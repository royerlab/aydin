How to implement supervised denoising using image translators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is quite easy to run supervised denoising with image translators provided
in Aydin API. You can see a quick example below using ImageTranslatorFGR:

.. code-block:: python

   from aydin.it.fgr import ImageTranslatorFGR
   from aydin.it.transforms.range import RangeTransform
   from aydin.it.transforms.padding import PaddingTransform

   it = ImageTranslatorFGR()
   it.add_transform(RangeTransform())
   it.add_transform(PaddingTransform())

   it.train(noisy, groundtruth)
   denoised = it.translate(noisy)


Similar to ImageTranslatorFGR implementation, same can be achieved with
ImageTranslatorCNNTorch as shown below:

.. code-block:: python

   from aydin.it.cnn_torch import ImageTranslatorCNNTorch
   from aydin.it.transforms.range import RangeTransform
   from aydin.it.transforms.padding import PaddingTransform

   it = ImageTranslatorCNNTorch()
   it.add_transform(RangeTransform())
   it.add_transform(PaddingTransform())

   it.train(noisy, groundtruth)
   denoised = it.translate(noisy)
